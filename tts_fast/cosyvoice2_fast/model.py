import os
import torch
from typing import List, Dict, AsyncGenerator
from hyperpyyaml import load_hyperpyyaml
from copy import deepcopy
from ruamel import yaml
import ray
from dataclasses import dataclass, field
import asyncio
from torch.nn import functional as F
import numpy as np
import enum
from loguru import logger
import uuid
from concurrent.futures import ThreadPoolExecutor
import time
import traceback
from ray.util import ActorPool

import matcha
import cosyvoice
import cosyvoice2_fast
from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
from cosyvoice.hifigan.generator import HiFTGenerator
from cosyvoice2_fast.frontend import CosyVoice2FrontEnd
from cosyvoice2_fast.llm import CosyVoice2LLMWrapper
from cosyvoice.utils.common import fade_in_out
from cosyvoice2_fast.utils import stream_context, set_flow_decoder, convert_onnx_to_trt


# 限制将 flow onnx 转换为 trt 时使用的 GPU 显存大小
ONNX2TRT_WORKSPACE_SIZE = int(os.getenv("ONNX2TRT_WORKSPACE_SIZE", 2))
# 根据 GPU 显存大小量及性能设置合适的 ESTIMATOR_COUNT
ESTIMATOR_COUNT = int(os.getenv("ESTIMATOR_COUNT", 1))
# flow model 使用的 gpu 大小
FLOW_USED_NUM_GPUS = float(os.getenv("FLOW_USED_NUM_GPUS", 0.15))
# hift model 使用的 gpu 大小
HIFT_USED_NUM_GPUS = float(os.getenv("HIFT_USED_NUM_GPUS", 0.05))

COMMON_OVERRIDES = {
    "__set_seed1": None,
    "__set_seed2": None,
    "__set_seed3": None,
    "__set_seed4": None,
    "qwen_pretrain_path": None,
    "llm": None,
    "flow": None,
    "hift": None,
    "parquet_opener": None,
    "get_tokenizer": None,
    "tokenize": None,
    "filter": None,
    "resample": None,
    "feat_extractor": None,
    "compute_fbank": None,
    "parse_embedding": None,
    "shuffle": None,
    "sort": None,
    "batch": None,
    "padding": None,
}

WAIT_INTERVAL = 1e-7
TIMEOUT = 30.0


def cache_output(generator_or_tensors: AsyncGenerator | torch.Tensor):
    if isinstance(generator_or_tensors, torch.Tensor):
        tensors = generator_or_tensors
        return tensors, {"cached": tensors.cpu()}

    generator = generator_or_tensors
    temp = {"cached": torch.empty((1, 0), device="cpu")}

    async def cached_generator():
        async for i in generator:
            yield i
            temp["cached"] = torch.cat([temp["cached"], i.cpu()], dim=1)

    return cached_generator(), temp


class ModelActor:
    def ready(self):
        return True


@ray.remote(num_gpus=FLOW_USED_NUM_GPUS, max_concurrency=100)
class FlowActor(ModelActor):
    def __init__(
        self, model_dir: str, overrides: dict, use_jit, use_trt, fp16, do_compile
    ):
        self.model_dir = model_dir
        self.overrides = deepcopy(overrides)
        self.overrides.pop("flow")
        self.use_jit = use_jit
        self.use_trt = use_trt
        self.fp16 = fp16
        self.do_compile = do_compile
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.flow_model = self.build()

    @ray.method(enable_task_events=False)
    @stream_context()
    def generate(
        self,
        speech_token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        speaker_embedding: torch.Tensor,
        finalized: bool,
        offset: int,
        id: str,
        index: int,
    ):
        try:
            with torch.amp.autocast(
                "cuda", torch.float16 if self.fp16 else torch.float
            ):
                speech_mel, _ = self.flow_model.inference(
                    token=speech_token.long().to(self.device),
                    token_len=torch.tensor(
                        [speech_token.shape[1]], dtype=torch.int32
                    ).to(self.device),
                    prompt_token=prompt_token.long().to(self.device),
                    prompt_token_len=torch.tensor(
                        [prompt_token.shape[1]], dtype=torch.int32
                    ).to(self.device),
                    prompt_feat=prompt_feat.to(self.device),
                    prompt_feat_len=torch.tensor(
                        [prompt_feat.shape[1]], dtype=torch.int32
                    ).to(self.device),
                    embedding=speaker_embedding.to(self.device),
                    streaming=not finalized,
                    finalize=finalized,
                )
            speech_mel = speech_mel[:, :, offset * self.flow_model.token_mel_ratio :]
            torch.cuda.empty_cache()
            return speech_mel, id, index
        except Exception as e:
            return e

    def build(self):
        with open(f"{self.model_dir}/cosyvoice2.yaml", "r") as f:
            ### create model
            flow: CausalMaskedDiffWithXvec = load_hyperpyyaml(
                f, overrides=self.overrides
            )["flow"]

            ### load weight
            flow.load_state_dict(
                torch.load(
                    f"{self.model_dir}/flow.pt",
                    weights_only=True,
                    map_location=self.device,
                ),
                strict=True,
            )

            ### fp16
            flow.fp16 = self.fp16
            dtype = "fp16" if self.fp16 else "fp32"
            if self.fp16:
                flow.half()

            # here fix flow encoder/decoder decoding_chunk_size
            # in the future will send it as arguments, or use cache
            flow.encoder.static_chunk_size = 2 * flow.input_frame_rate
            flow.decoder.estimator.static_chunk_size = (
                2 * flow.input_frame_rate * flow.token_mel_ratio
            )

            ### encoder jit
            if self.use_jit:
                jit_encoder_path = f"{self.model_dir}/flow.encoder.{dtype}.zip"
                jit_encoder = torch.jit.load(jit_encoder_path, map_location=self.device)
                flow.encoder = jit_encoder

            ### decoder trt
            if self.use_trt:
                decoder_estimator_path = (
                    f"{self.model_dir}/flow.decoder.estimator.{dtype}.mygpu.plan"
                )
                decoder_onnx_path = f"{self.model_dir}/flow.decoder.estimator.fp32.onnx"
                if not os.path.exists(decoder_estimator_path):
                    convert_onnx_to_trt(
                        decoder_estimator_path,
                        decoder_onnx_path,
                        self.fp16,
                        ONNX2TRT_WORKSPACE_SIZE,
                    )
                if os.path.getsize(decoder_estimator_path) == 0:
                    raise ValueError(
                        f"{decoder_estimator_path} is empty file, delete it and export again!"
                    )
                del flow.decoder.estimator

                import tensorrt as trt

                with open(decoder_estimator_path, "rb") as f:
                    flow.decoder.estimator_engine = trt.Runtime(
                        trt.Logger(trt.Logger.INFO)
                    ).deserialize_cuda_engine(f.read())
                if flow.decoder.estimator_engine is None:
                    raise ValueError(f"failed to load trt {decoder_estimator_path}")

                set_flow_decoder(flow.decoder, ESTIMATOR_COUNT)

            flow.to(self.device).eval()
            if self.do_compile:
                flow = torch.compile(flow)
            return flow


@ray.remote(num_gpus=HIFT_USED_NUM_GPUS, max_concurrency=100)
class HiftActor(ModelActor):
    def __init__(self, model_dir: str, overrides: dict, do_compile):
        self.model_dir = model_dir
        self.overrides = deepcopy(overrides)
        self.overrides.pop("hift")
        self.do_compile = do_compile
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.hift_model = self.build()

    @ray.method(enable_task_events=False)
    def generate(self, speech_mel, cache_source, id, index):
        try:
            speech_pcm, new_source = self.hift_model.inference(
                speech_feat=speech_mel.to(self.device),
                cache_source=cache_source.to(self.device),
            )
            torch.cuda.empty_cache()
            return speech_pcm, new_source, id, index
        except Exception as e:
            return e

    def build(self):
        with open(f"{self.model_dir}/cosyvoice2.yaml", "r") as f:
            ### create model
            hift: HiFTGenerator = load_hyperpyyaml(f, overrides=self.overrides)["hift"]

            ### load weight
            sd = {
                k.replace("generator.", ""): v
                for k, v in torch.load(
                    f"{self.model_dir}/hift.pt",
                    weights_only=True,
                    map_location=self.device,
                ).items()
            }
            hift.load_state_dict(sd, strict=True)
            hift.to(self.device).eval()
            if self.do_compile:
                hift = torch.compile(hift)
            return hift


@dataclass
class Request:
    id: str

    # text -> token
    tokens: List = field(default_factory=list)
    # token -> mel
    mels: List[torch.Tensor] = field(default_factory=lambda: [torch.empty(1, 80, 0)])
    # mel -> source
    sources: List[torch.Tensor] = field(default_factory=lambda: [torch.zeros(1, 1, 0)])
    # mel -> pcm
    pcms: List[torch.Tensor | None] = field(default_factory=lambda: [None])

    # status related
    llm_done: bool = False
    end_index: int = -1
    stop: bool = False


class Scheduler:
    def __init__(
        self,
        model_dir,
        flow_count=1,
        hift_count=1,
        use_jit=False,
        use_trt=False,
        do_compile=True,
        fp16=False,
    ):
        ray.init(runtime_env={"py_modules": [matcha, cosyvoice, cosyvoice2_fast]})
        self.set_config(model_dir)

        self.llm = CosyVoice2LLMWrapper(model_dir)
        logger.info("LLM is ready.")

        flow_actors = [
            FlowActor.remote(
                model_dir, COMMON_OVERRIDES, use_jit, use_trt, fp16, do_compile
            )
            for _ in range(flow_count)
        ]
        if all(ray.get([a.ready.remote() for a in flow_actors])):
            logger.info("Flow actors are ready.")
        self.flow_pool = ActorPool(flow_actors)

        hift_actors = [
            HiftActor.remote(model_dir, COMMON_OVERRIDES, do_compile)
            for _ in range(hift_count)
        ]
        if all(ray.get([a.ready.remote() for a in hift_actors])):
            logger.info("Hift actors are ready.")
        self.hift_pool = ActorPool(hift_actors)

        self.thread_pool = ThreadPoolExecutor()
        self.thread_pool.submit(self.get_flow_outputs)
        self.thread_pool.submit(self.get_hift_outputs)
        self.cache: Dict[str, Request] = {}

    def set_config(self, model_dir):
        with open(f"{model_dir}/cosyvoice2.yaml", "r") as f:
            cfg = yaml.YAML().load(f)
            self.hop_len: int = cfg["token_frame_rate"] * 2
            self.token_mel_ratio: int = cfg["token_mel_ratio"]
            self.pre_lookahead_len: int = cfg["flow"]["pre_lookahead_len"]
            self.sample_rate: int = cfg["sample_rate"]
            self.mel_cache_len = 8
            self.source_cache_len = int(self.mel_cache_len * 480)
            self.speech_window = np.hamming(2 * self.source_cache_len)

    def get_flow_outputs(self):
        while True:
            try:
                if self.flow_pool.has_next():
                    res = self.flow_pool.get_next_unordered()
                    if not isinstance(res, Exception):
                        speech_mel, id, index = res
                        if id in self.cache:
                            self.cache[id].mels[index] = speech_mel
                    else:
                        raise res
                else:
                    time.sleep(WAIT_INTERVAL)
            except:
                logger.error(f"Error in getting flow: {traceback.format_exc()}")

    def get_hift_outputs(self):
        while True:
            try:
                if self.hift_pool.has_next():
                    res = self.hift_pool.get_next_unordered()
                    if not isinstance(res, Exception):
                        tts_pcm, tts_source, id, index = res
                        if id in self.cache:
                            req = self.cache[id]
                            req.sources[index] = tts_source
                            if req.pcms[index - 1] is not None:
                                tts_pcm = fade_in_out(
                                    tts_pcm,
                                    req.pcms[index - 1][:, -self.source_cache_len :],
                                    self.speech_window,
                                )
                            req.pcms[index] = tts_pcm
                    else:
                        raise res
                else:
                    time.sleep(WAIT_INTERVAL)
            except:
                logger.error(f"Error in getting hift: {traceback.format_exc()}")

    def submit_flow_inputs(self, req: Request, offset, flow_length, finalized):
        this_speech_tokens = torch.tensor(req.tokens[:flow_length])
        this_speech_tokens = this_speech_tokens.unsqueeze(dim=0)

        req.mels.append(None)
        req.sources.append(None)
        req.pcms.append(None)
        curr_idx = len(req.mels) - 1
        if finalized:
            req.end_index = curr_idx

        self.flow_pool.submit(
            lambda actor, args: actor.generate.remote(*args),
            (
                this_speech_tokens,
                req.flow_prompt_speech_tokens,
                req.prompt_speech_feature,
                req.speaker_embeddings,
                finalized,
                offset,
                req.id,
                curr_idx,
            ),
        )

    def submit_hift_inputs(self, req: Request, index):
        last_index = index - 1
        this_mel = req.mels[index]
        last_mel = req.mels[last_index][:, :, -self.mel_cache_len :]
        this_mel = torch.concat([last_mel.to(this_mel.device), this_mel], dim=2)
        last_source = req.sources[last_index][:, :, -self.source_cache_len :]

        if not req.stream and req.speed != 1.0:
            this_mel = F.interpolate(
                this_mel, size=int(this_mel.shape[2] / req.speed), mode="linear"
            )

        self.hift_pool.submit(
            lambda actor, args: actor.generate.remote(*args),
            (this_mel, last_source, req.id, index),
        )

    async def llm_job(
        self,
        req: Request,
        keep_original_prompt=False,
        min_prefix_count=1,
        max_length=512,
    ):
        try:
            ptt_orig = req.prompt_text_tokens.cpu()
            pst_orig = req.llm_prompt_speech_tokens.cpu()
            curr_count = 0
            curr_len = []
            prefix_ptt = []
            prefix_pst = []
            last_speech_tokens = []

            async for tts_text_tokens in req.input_generator:
                if len(prefix_ptt) > 0:
                    if keep_original_prompt:
                        ptt_input = torch.cat([ptt_orig] + prefix_ptt, dim=1)
                        pst_input = torch.cat([pst_orig] + prefix_pst, dim=1)
                    else:
                        ptt_input = torch.cat(prefix_ptt, dim=1)
                        pst_input = torch.cat(prefix_pst, dim=1)
                else:
                    ptt_input = ptt_orig
                    pst_input = pst_orig

                last_speech_tokens.clear()
                tt_input, temp = cache_output(tts_text_tokens)
                output_generator = self.llm(tt_input, ptt_input, pst_input, req.id)
                async for speech_token in output_generator:
                    req.tokens.append(speech_token)
                    last_speech_tokens.append(speech_token)

                curr_count += 1
                curr_len.append(temp["cached"].shape[1] + len(last_speech_tokens))
                prefix_ptt.append(temp["cached"])
                prefix_pst.append(torch.tensor([last_speech_tokens]).cpu())

                while curr_count > min_prefix_count and sum(curr_len) > max_length:
                    curr_count -= 1
                    curr_len.pop(0)
                    prefix_ptt.pop(0)
                    prefix_pst.pop(0)

        except BaseException as e:
            req.stop = True
            if not isinstance(e, asyncio.CancelledError):
                logger.error(f"Error in llm job: {traceback.format_exc()}")
        finally:
            req.llm_done = True

    def flow_job(self, req: Request):
        try:
            offset = 0
            __start = time.time()
            while not req.stop:
                if req.llm_done:
                    flow_length = len(req.tokens)
                    if flow_length > 0:
                        self.submit_flow_inputs(req, offset, flow_length, True)
                    break
                elif req.stream:
                    flow_length = offset + self.hop_len + self.pre_lookahead_len
                    if len(req.tokens) >= flow_length:
                        # generated length, length to generate, finalized
                        self.submit_flow_inputs(req, offset, flow_length, False)
                        offset += self.hop_len
                        __start = time.time()
                    else:
                        if time.time() - __start > TIMEOUT:
                            raise TimeoutError("The flow job is timeout.")
                        time.sleep(WAIT_INTERVAL)
        except:
            req.stop = True
            logger.error(f"Error in flow job: {traceback.format_exc()}")

    def hift_job(self, req: Request):
        try:
            index = 1
            __start = time.time()
            while not req.stop:
                if (
                    len(req.mels) > index
                    and req.mels[index] is not None
                    and req.sources[index - 1] is not None
                ):
                    self.submit_hift_inputs(req, index)
                    if index == req.end_index:
                        break
                    index += 1
                    __start = time.time()
                else:
                    if time.time() - __start > TIMEOUT:
                        raise TimeoutError("The hift job is timeout.")
                    time.sleep(WAIT_INTERVAL)
        except:
            req.stop = True
            logger.error(f"Error in hift job: {traceback.format_exc()}")

    async def request(
        self,
        request_id,
        input_generator,
        prompt_text_tokens,
        llm_prompt_speech_tokens,
        speaker_embeddings,
        flow_prompt_speech_tokens,
        prompt_speech_feature,
        stream,
        speed,
    ):
        req = Request(request_id)
        req.input_generator = input_generator
        req.prompt_text_tokens = prompt_text_tokens
        req.llm_prompt_speech_tokens = llm_prompt_speech_tokens
        req.speaker_embeddings = speaker_embeddings
        req.flow_prompt_speech_tokens = flow_prompt_speech_tokens
        req.prompt_speech_feature = prompt_speech_feature
        req.stream = stream
        req.speed = speed

        self.cache[request_id] = req

        llm_job = asyncio.create_task(self.llm_job(req))
        self.thread_pool.submit(self.hift_job, req)
        self.thread_pool.submit(self.flow_job, req)

        try:
            index = 1
            __start = time.time()
            while not req.stop:
                if len(req.pcms) > index and req.pcms[index] is not None:
                    finalized = index == req.end_index
                    if finalized:
                        yield req.pcms[index].cpu().numpy().flatten()
                        break
                    else:
                        yield (
                            req.pcms[index][:, : -self.source_cache_len]
                            .cpu()
                            .numpy()
                            .flatten()
                        )
                    index += 1
                    __start = time.time()
                else:
                    if time.time() - __start > TIMEOUT:
                        raise TimeoutError("Waiting pcm is timeout.")
                    await asyncio.sleep(WAIT_INTERVAL)
        finally:
            req.stop = True
            if not llm_job.done():
                llm_job.cancel()
            del self.cache[request_id]


def load_frontend(model_dir: str, overrides: dict):
    with open(f"{model_dir}/cosyvoice2.yaml", "r") as f:
        overrides = deepcopy(overrides)
        overrides["qwen_pretrain_path"] = os.path.join(model_dir, "CosyVoice-BlankEN")
        overrides.pop("get_tokenizer")
        overrides.pop("feat_extractor")
        cfg = load_hyperpyyaml(f, overrides=overrides)
        frontend = CosyVoice2FrontEnd(
            cfg["get_tokenizer"],
            cfg["feat_extractor"],
            f"{model_dir}/campplus.onnx",
            f"{model_dir}/speech_tokenizer_v2.onnx",
            cfg["allowed_special"],
        )
        return frontend


class CosyVoiceInputType(enum.Enum):
    SINGLE = enum.auto()
    GENERATOR = enum.auto()
    QUEUE = enum.auto()


class CosyVoice2:
    def __init__(
        self,
        model_dir,
        flow_count=1,
        hift_count=1,
        use_jit=False,
        use_trt=True,
        do_compile=True,
        fp16=True,
    ):
        self.scheduler = Scheduler(
            model_dir, flow_count, hift_count, use_jit, use_trt, do_compile, fp16
        )
        self.sample_rate = self.scheduler.sample_rate
        self.frontend = load_frontend(model_dir, COMMON_OVERRIDES)

    async def preprocess(self, text, split_text):
        res = self.frontend.text_normalize(text, split_text, True)
        if res is None:
            res = []
        if not isinstance(res, list):
            res = [res]
        for i in res:
            yield self.frontend.extract_text_token(i)

    async def wrap_to_generator(
        self, obj, split_text, input_type=CosyVoiceInputType.SINGLE
    ):
        if input_type == CosyVoiceInputType.SINGLE:
            # normalized -> str, Generator, AsyncGenerator
            async for normalized in self.preprocess(obj, split_text):
                yield normalized
        elif input_type == CosyVoiceInputType.GENERATOR:
            async for i in obj:
                async for j in self.wrap_to_generator(i, split_text):
                    yield j
        elif input_type == CosyVoiceInputType.QUEUE:
            while True:
                i = await obj.get()
                if i is None:
                    break
                async for j in self.wrap_to_generator(i, split_text):
                    yield j

    def async_tts(
        self,
        id,
        text,
        flow_embedding,
        prompt_text=torch.zeros(1, 0, dtype=torch.int32),
        llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat=torch.zeros(1, 0, 80),
        split_text=False,
        stream=False,
        speed=1.0,
        input_type=CosyVoiceInputType.SINGLE,
    ):
        req_args = {
            "request_id": id or uuid.uuid4().hex[:7],
            "input_generator": self.wrap_to_generator(text, split_text, input_type),
            "prompt_text_tokens": prompt_text,
            "llm_prompt_speech_tokens": llm_prompt_speech_token,
            "speaker_embeddings": flow_embedding,
            "flow_prompt_speech_tokens": flow_prompt_speech_token,
            "prompt_speech_feature": prompt_speech_feat,
            "stream": stream,
            "speed": speed,
        }
        return self.scheduler.request(**req_args)
