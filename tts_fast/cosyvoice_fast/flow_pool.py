import uuid
import os
import torch
import ray
import asyncio
from loguru import logger
import traceback
from ray.util import ActorPool
from omegaconf import DictConfig
import gc
import numpy as np

from .utils import (
    get_flow_decoder_estimator_input_shapes,
    convert_onnx_to_trt,
    set_flow_decoder_estimator,
    simplify_onnx,
    slim_onnx,
)
from .common import (
    VERSION,
    FLOW_ACTOR_NUM_GPUS,
    FLOW_ACTOR_COUNT,
    FLOW_DTYPE,
    FLOW_JIT,
    FLOW_TRT,
    FLOW_COMPILE,
    ACTOR_MAX_CONCURRENCY,
    Prompt,
    Params,
    ONNX2TRT_WORKSPACE_SIZE,
    COMPILATION_CACHE_DIR,
    FLOW_ESTIMATOR_COUNT,
)


class FlowActor:
    def __init__(self, model_dir, version, use_jit, use_trt, dtype, do_compile, cache_dir):
        self.model_dir = model_dir
        self.version = version
        self.use_jit = use_jit
        self.use_trt = use_trt
        self.dtype = dtype
        self.torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        self.do_compile = do_compile
        self.cache_dir = cache_dir
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.flow = self.build()

    def ready(self):
        return True

    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()

    @ray.method(enable_task_events=False, num_returns=3)
    @torch.inference_mode()
    def generate(self, ret_flags, *infer_args):
        try:
            if self.version in ["cosyvoice2", "cosyvoice3"]:
                res = self.generate_cosyvoice(*infer_args)
            elif self.version == "cosyvoice2_stepaudio_whole":
                res = self.generate_stepaudio_whole(*infer_args)
            else:
                res = self.generate_stepaudio_stream(*infer_args)
            return ret_flags, res, None
        except BaseException:
            return ret_flags, None, traceback.format_exc()

    def generate_cosyvoice(
        self,
        speech_tokens: torch.Tensor,
        prompt_speech_tokens: torch.Tensor,
        prompt_speech_mels: torch.Tensor,
        speaker_embedding: torch.Tensor,
        finalized: bool,
        offset: int,
        speaker_id: str,
        request_id: str,
    ):
        with torch.amp.autocast("cuda", self.torch_dtype):
            speech_mels, _ = self.flow.inference(
                token=speech_tokens.to(self.device),
                token_len=torch.tensor([speech_tokens.shape[1]], dtype=torch.int32).to(self.device),
                prompt_token=prompt_speech_tokens.to(self.device),
                prompt_token_len=torch.tensor([prompt_speech_tokens.shape[1]], dtype=torch.int32).to(self.device),
                prompt_feat=prompt_speech_mels.to(self.device),
                prompt_feat_len=torch.tensor([prompt_speech_mels.shape[1]], dtype=torch.int32).to(self.device),
                embedding=speaker_embedding.to(self.device),
                streaming=not finalized,
                finalize=finalized,
            )
        speech_mels = speech_mels[:, :, offset * self.flow.token_mel_ratio :]
        speech_mels = speech_mels.cpu().share_memory_()
        return speech_mels

    def generate_stepaudio_whole(
        self,
        speech_tokens: torch.Tensor,
        prompt_speech_tokens: torch.Tensor,
        prompt_speech_mels: torch.Tensor,
        speaker_embedding: torch.Tensor,
        finalized: bool,
        offset: int,
        speaker_id: str,
        request_id: str,
    ):
        with torch.amp.autocast("cuda", self.torch_dtype):
            input_tokens = torch.cat([prompt_speech_tokens, speech_tokens], dim=1)
            input_lens = torch.tensor([input_tokens.shape[1]])
            prompt_speech_mels_lens = torch.tensor([prompt_speech_mels.shape[1]])
            mels, mels_lens = self.flow.inference(
                input_tokens.to(self.device),
                input_lens.to(self.device),
                prompt_speech_mels.to(self.device),
                prompt_speech_mels_lens.to(self.device),
                speaker_embedding.to(self.device),
                n_timesteps=10,
            )

        si = prompt_speech_mels_lens[0].item() + (offset * self.up_scale_factor)
        ei = mels_lens[0].item() - self.pre_lookahead_len * self.up_scale_factor
        speech_mels = mels[:, :, si:ei]
        speech_mels = speech_mels.cpu().share_memory_()
        return speech_mels

    def generate_stepaudio_stream(
        self,
        speech_tokens: torch.Tensor,
        prompt_speech_tokens: torch.Tensor,
        prompt_speech_mels: torch.Tensor,
        speaker_embedding: torch.Tensor,
        finalized: bool,
        offset: int,
        speaker_id: str,
        request_id: str,
    ):
        if speaker_id not in self.speaker_cache:
            pre_lookahead = prompt_speech_tokens[:, : self.pre_lookahead_len]
            this_pst = torch.cat([prompt_speech_tokens, pre_lookahead], dim=1)
            with torch.amp.autocast("cuda", self.torch_dtype):
                this_speaker_cache = self.flow.setup_cache(
                    this_pst.to(self.device),
                    prompt_speech_mels.to(self.device),
                    speaker_embedding.to(self.device),
                    n_timesteps=10,
                )
            self.speaker_cache[speaker_id] = this_speaker_cache

        if request_id not in self.request_cache:
            self.request_cache[request_id] = {k: v.clone() for k, v in self.speaker_cache[speaker_id].items()}

        inference_cache = self.request_cache[request_id]
        with torch.amp.autocast("cuda", self.torch_dtype):
            speech_mel, new_cache = self.flow.inference_chunk(
                token=speech_tokens.to(self.device),
                spk=speaker_embedding.to(self.device),
                cache=inference_cache,
                last_chunk=finalized,
                n_timesteps=10,
            )

        if not finalized:
            psm_len = prompt_speech_mels.shape[1]
            if new_cache["estimator_att_cache"].shape[4] > (psm_len + 100):
                new_cache["estimator_att_cache"] = torch.cat(
                    [
                        new_cache["estimator_att_cache"][:, :, :, :, :psm_len],
                        new_cache["estimator_att_cache"][:, :, :, :, -100:],
                    ],
                    dim=4,
                )
            self.request_cache[request_id] = new_cache
        else:
            self.request_cache.pop(request_id)

        speech_mel = speech_mel.float().cpu().share_memory_()
        return speech_mel

    def create_model_cosyvoice2(self):
        from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
        from cosyvoice.transformer.upsample_encoder import UpsampleConformerEncoder
        from cosyvoice.flow.flow_matching import CausalConditionalCFM
        from cosyvoice.flow.decoder import CausalConditionalDecoder

        flow = CausalMaskedDiffWithXvec(
            input_size=512,
            output_size=80,
            spk_embed_dim=192,
            output_type="mel",
            vocab_size=6561,
            input_frame_rate=25,
            only_mask_loss=True,
            token_mel_ratio=2,
            pre_lookahead_len=3,
            encoder=UpsampleConformerEncoder(
                output_size=512,
                attention_heads=8,
                linear_units=2048,
                num_blocks=6,
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                attention_dropout_rate=0.1,
                normalize_before=True,
                input_layer="linear",
                pos_enc_layer_type="rel_pos_espnet",
                selfattention_layer_type="rel_selfattn",
                input_size=512,
                use_cnn_module=False,
                macaron_style=False,
                static_chunk_size=25,
            ),
            decoder=CausalConditionalCFM(
                in_channels=240,
                n_spks=1,
                spk_emb_dim=80,
                cfm_params=DictConfig(
                    content={
                        "sigma_min": 1e-06,
                        "solver": "euler",
                        "t_scheduler": "cosine",
                        "training_cfg_rate": 0.2,
                        "inference_cfg_rate": 0.7,
                        "reg_loss_type": "l1",
                    }
                ),
                estimator=CausalConditionalDecoder(
                    in_channels=320,
                    out_channels=80,
                    channels=[256],
                    dropout=0.0,
                    attention_head_dim=64,
                    n_blocks=4,
                    num_mid_blocks=12,
                    num_heads=8,
                    act_fn="gelu",
                    static_chunk_size=25 * 2,
                    num_decoding_left_chunks=-1,
                ),
            ),
        )
        flow.encoder.static_chunk_size = 2 * flow.input_frame_rate
        flow.decoder.estimator.static_chunk_size = 2 * flow.input_frame_rate * flow.token_mel_ratio
        return flow

    def create_model_cosyvoice3(self):
        from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
        from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer
        from cosyvoice.flow.flow_matching import CausalConditionalCFM
        from cosyvoice.flow.DiT.dit import DiT

        flow = CausalMaskedDiffWithDiT(
            input_size=80,
            output_size=80,
            spk_embed_dim=192,
            output_type="mel",
            vocab_size=6561,
            input_frame_rate=25,
            only_mask_loss=True,
            token_mel_ratio=2,
            pre_lookahead_len=3,
            pre_lookahead_layer=PreLookaheadLayer(in_channels=80, channels=1024, pre_lookahead_len=3),
            decoder=CausalConditionalCFM(
                in_channels=240,
                n_spks=1,
                spk_emb_dim=80,
                cfm_params=DictConfig(
                    content={
                        "sigma_min": 1e-06,
                        "solver": "euler",
                        "t_scheduler": "cosine",
                        "training_cfg_rate": 0.2,
                        "inference_cfg_rate": 0.7,
                        "reg_loss_type": "l1",
                    }
                ),
                estimator=DiT(
                    dim=1024,
                    depth=22,
                    heads=16,
                    dim_head=64,
                    ff_mult=2,
                    mel_dim=80,
                    mu_dim=80,
                    spk_dim=80,
                    out_channels=80,
                    static_chunk_size=25 * 2,
                    num_decoding_left_chunks=-1,
                ),
            ),
        )
        return flow

    def create_model_stepaudio(self):
        from stepaudio.flow.flow import CausalMaskedDiffWithXvec
        from stepaudio.transformer.upsample_encoder_v2 import UpsampleConformerEncoderV2
        from stepaudio.flow.flow_matching import CausalConditionalCFM
        from stepaudio.flow.decoder_dit import DiT

        self.up_scale_factor = 2
        self.pre_lookahead_len = 3

        flow = CausalMaskedDiffWithXvec(
            input_size=512,
            output_size=80,
            spk_embed_dim=192,
            output_type="mel",
            vocab_size=6561,
            encoder=UpsampleConformerEncoderV2(
                input_size=512,
                output_size=512,
                input_layer="linear",
                pre_lookahead_len=self.pre_lookahead_len,
                num_blocks=6,
                num_up_blocks=4,
                up_stride=2,
                up_scale_factor=self.up_scale_factor,
                attention_heads=8,
                pos_enc_layer_type="rel_pos_espnet",
                selfattention_layer_type="rel_selfattn",
                key_bias=True,
                linear_units=2048,
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                attention_dropout_rate=0.1,
                normalize_before=True,
            ),
            decoder=CausalConditionalCFM(
                inference_cfg_rate=0.7,
                estimator=DiT(
                    in_channels=320,
                    out_channels=80,
                    mlp_ratio=4.0,
                    depth=16,
                    num_heads=8,
                    head_dim=64,
                    hidden_size=512,
                ),
            ),
        )

        return flow

    def build(self):
        ### create flow model
        if self.version == "cosyvoice2":
            flow = self.create_model_cosyvoice2()
        elif self.version == "cosyvoice3":
            flow = self.create_model_cosyvoice3()
        elif self.version == "cosyvoice2_stepaudio_whole":
            flow = self.create_model_stepaudio()
        else:
            flow = self.create_model_stepaudio()
            self.speaker_cache = {}
            self.request_cache = {}

        ### load weights
        sd = torch.load(f"{self.model_dir}/flow.pt", map_location="cpu", weights_only=True)
        flow.load_state_dict(sd, strict=True)

        ### use jit to accelerate flow encoder
        if self.use_jit:
            jit_encoder_path = f"{self.model_dir}/flow.encoder.{self.dtype}.zip"
            jit_encoder = torch.jit.load(jit_encoder_path, map_location=self.device)
            flow.encoder = jit_encoder

        ### use trt to accelerate flow decoder estimator
        if self.use_trt:
            import tensorrt as trt

            prefix = f"{self.model_dir}/flow.decoder.estimator"
            trt_path = f"{prefix}.{self.version}.{self.dtype}.plan"
            onnx_path = f"{prefix}.fp32.onnx" if self.version != "cosyvoice2_stepaudio_stream" else f"{prefix}.fp32.stream.onnx"

            if not os.path.exists(trt_path):
                # optimize onnx model
                onnx_path = slim_onnx(simplify_onnx(onnx_path))

                if self.version == "cosyvoice3" and self.dtype == "fp16":

                    def set_layer_precision(layer):
                        if not layer.name.endswith(("attn/MatMul_1", "attn/MatMul")):
                            layer.precision = trt.DataType.FLOAT
                            for i in range(layer.num_outputs):
                                layer.set_output_type(i, trt.DataType.FLOAT)
                else:
                    set_layer_precision = None

                input_shapes = get_flow_decoder_estimator_input_shapes(self.version)

                convert_onnx_to_trt(
                    onnx_path,
                    trt_path,
                    input_shapes,
                    self.dtype,
                    workspace_size=ONNX2TRT_WORKSPACE_SIZE,
                    optimization_level=3,
                    timing_cache_path=os.path.join(self.cache_dir, "trt_timing.cache"),
                    set_layer_precision=set_layer_precision,
                )

            with open(trt_path, "rb") as f:
                estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
            if estimator_engine is None:
                raise ValueError(f"Failed to load trt {trt_path}.")

            set_flow_decoder_estimator(flow, estimator_engine, self.device, FLOW_ESTIMATOR_COUNT)

        ### convert model dtype
        if self.dtype == "fp16":
            flow.half()
        elif self.dtype == "bf16":
            flow.bfloat16()

        ### move to cuda and eval mode
        flow.to(self.device).eval()

        ### compile flow model
        if self.do_compile:
            flow = torch.compile(flow, fullgraph=True, dynamic=True, mode="max-autotune")

        return flow


class FlowPool:
    def __init__(
        self,
        model_dir,
        pre_lookahead_len: int,
        token_frame_rate: int,
    ):
        FlowActor_class = ray.remote(FlowActor).options(num_gpus=FLOW_ACTOR_NUM_GPUS, max_concurrency=ACTOR_MAX_CONCURRENCY)
        actors = [
            FlowActor_class.remote(
                model_dir,
                VERSION,
                FLOW_JIT,
                FLOW_TRT,
                FLOW_DTYPE,
                FLOW_COMPILE,
                COMPILATION_CACHE_DIR,
            )
            for _ in range(FLOW_ACTOR_COUNT)
        ]
        if all(ray.get([a.ready.remote() for a in actors])):
            logger.info(f"{self.__class__.__name__} is ready.")
        self.actors = actors
        self.pool = ActorPool(actors)

        self.pre_lookahead_len = pre_lookahead_len
        if VERSION == "cosyvoice2_stepaudio_stream":
            self.hop_len = token_frame_rate
            self.handle_input = self.handle_input_stepaudio_stream
        else:
            self.hop_len = token_frame_rate * 2
            self.handle_input = self.handle_input_default

        self.buffer: dict[str, asyncio.PriorityQueue] = {}
        asyncio.create_task(self.handle_output())

    async def handle_output(self):
        while True:
            while self.pool.has_next():
                try:
                    flags, res, error = self.pool.get_next_unordered(timeout=0)
                    id, index, finalized = flags
                    if id in self.buffer:
                        await self.buffer[id].put((index, finalized, res, error))
                except TimeoutError:
                    await asyncio.sleep(0)
            await asyncio.sleep(0)

    def submit(self, ret_args, *infer_args):
        self.pool.submit(
            lambda actor, args: actor.generate.remote(args[0], *args[1:]),
            (ret_args, *infer_args),
        )

    def clean(self):
        for a in self.actors:
            a.clean.remote()

    async def handle_input_default(
        self,
        id,
        input_queue: asyncio.Queue,
        speech_tokens,
        speech_mels,
        speaker_embedding,
        speaker_id,
        params: Params,
    ):
        first_hop_padding = int(np.ceil(speech_tokens.shape[1] / self.hop_len) * self.hop_len - speech_tokens.shape[1])
        pre_lookahead_len = self.pre_lookahead_len
        flow_window_size = max(params.flow_window_size, params.flow_window_shift)
        flow_window_shift = params.flow_window_shift

        speech_tokens = ray.put(speech_tokens)
        speech_mels = ray.put(speech_mels)
        speaker_embedding = ray.put(speaker_embedding)

        offset = 0
        index = 0
        received = []
        while True:
            speech_token = await input_queue.get()
            if speech_token is None:
                break
            received.append(speech_token)

            if params.stream:
                hop_len = self.hop_len + first_hop_padding if offset == 0 else self.hop_len

                if flow_window_shift > 0:
                    while offset > flow_window_size:
                        offset -= flow_window_shift
                        received = received[flow_window_shift:]

                flow_len = offset + hop_len + pre_lookahead_len
                if len(received) > flow_len:  # no `=` ensure remaning
                    input_tokens = torch.tensor(received[:flow_len]).unsqueeze(0)
                    self.submit(
                        (id, index, False),
                        input_tokens,
                        speech_tokens,
                        speech_mels,
                        speaker_embedding,
                        False,
                        offset,
                        speaker_id,
                        id,
                    )
                    offset += hop_len
                    index += 1

        if len(received) > 0:
            input_tokens = torch.tensor(received[:]).unsqueeze(0)
            self.submit(
                (id, index, True),
                input_tokens,
                speech_tokens,
                speech_mels,
                speaker_embedding,
                True,
                offset,
                speaker_id,
                id,
            )

    async def handle_input_stepaudio_stream(
        self,
        id,
        input_queue: asyncio.Queue,
        speech_tokens,
        speech_mels,
        speaker_embedding,
        speaker_id,
        params: Params,
    ):
        speech_tokens = ray.put(speech_tokens)
        speech_mels = ray.put(speech_mels)
        speaker_embedding = ray.put(speaker_embedding)

        index = 0
        received = []
        flow_len = self.hop_len + self.pre_lookahead_len
        while True:
            speech_token = await input_queue.get()
            if speech_token is None:
                break
            received.append(speech_token)

            if params.stream:
                if len(received) > flow_len:
                    input_tokens = torch.tensor(received[:flow_len]).unsqueeze(0)
                    self.submit(
                        (id, index, False),
                        input_tokens,
                        speech_tokens,
                        speech_mels,
                        speaker_embedding,
                        False,
                        None,
                        speaker_id,
                        id,
                    )
                    received = received[self.hop_len :]
                    index += 1

        if len(received) > 0:
            input_tokens = torch.tensor(received[:]).unsqueeze(0)
            self.submit(
                (id, index, True),
                input_tokens,
                speech_tokens,
                speech_mels,
                speaker_embedding,
                True,
                None,
                speaker_id,
                id,
            )

    async def run(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue, prompt: Prompt, params: Params):
        id = uuid.uuid4().hex
        self.buffer[id] = asyncio.PriorityQueue()

        speech_tokens = prompt.flow_speech_tokens.detach()
        speech_mels = prompt.speech_mels.detach()
        speaker_embedding = prompt.speaker_embedding.detach()
        speaker_id = prompt.speaker_id

        handle_input_task = asyncio.create_task(
            self.handle_input(id, input_queue, speech_tokens, speech_mels, speaker_embedding, speaker_id, params)
        )

        try:
            expected_index = 0
            while True:
                index, finalized, res, error = await self.buffer[id].get()
                if error:
                    raise Exception(error)
                speech_mel = res
                if index == expected_index:
                    await output_queue.put((finalized, speech_mel))
                    if finalized:
                        break
                    expected_index += 1
                else:
                    await self.buffer[id].put((index, finalized, speech_mel, None))
                    await asyncio.sleep(0)
        finally:
            self.clean()
            await output_queue.put(None)
            handle_input_task.cancel()
            del self.buffer[id]
