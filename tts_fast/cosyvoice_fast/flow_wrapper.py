import threading
from collections.abc import AsyncGenerator
import os
import torch
import ray
import asyncio
import traceback
from ray import serve
from ray.serve.handle import DeploymentHandle
from omegaconf import DictConfig
import gc
import numpy as np
from collections import OrderedDict

from .common import Prompt, Params
from .utils import CudaCacheCleaner


class PromptCacheDict:
    def __init__(self, max_size):
        self.pinned = OrderedDict()
        self.max_size = max_size

    def put(self, id, data):
        self.pinned[id] = data
        self.pinned.move_to_end(id, last=False)

        if len(self.pinned) > self.max_size:
            self.pinned.popitem(last=True)
            gc.collect()
            torch.cuda.empty_cache()

    def get(self, id):
        if id in self.pinned:
            self.pinned.move_to_end(id, last=False)
            return self.pinned[id]
        return None


@serve.deployment
class FlowActor:
    def __init__(
        self,
        model_dir: str,
        version: str,
        use_jit,
        use_trt,
        dtype,
        trt_workspace_size,
        num_flow_estimators,
        do_compile,
        compilation_cache_dir,
        prompt_cache_size,
        use_cuda_cache_cleaner,
        cuda_cache_clean_delay,
        cuda_cache_clean_interval,
    ):
        self.model_dir = model_dir
        self.version = version
        self.use_jit = use_jit
        self.use_trt = use_trt
        self.dtype = dtype
        self.torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        self.trt_workspace_size = trt_workspace_size
        self.num_flow_estimators = num_flow_estimators
        self.do_compile = do_compile
        self.compilation_cache_dir = compilation_cache_dir
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.prompt_cache_size = prompt_cache_size
        self.flow = self.build()
        self.lock = threading.Lock()
        self.cuda_cache_cleaner = CudaCacheCleaner(
            cuda_cache_clean_delay, cuda_cache_clean_interval, device=self.device, enabled=use_cuda_cache_cleaner
        )

    def generate(
        self,
        speech_tokens,
        prompt_speech_tokens,
        prompt_speech_mels,
        speaker_embedding,
        finalized,
        offset,
        speaker_id,
        request_id,
    ):
        try:
            with self.lock, self.cuda_cache_cleaner.track():
                if self.version in ["cosyvoice2", "cosyvoice3"]:
                    res = self.generate_cv2_cv3(speech_tokens, prompt_speech_tokens, prompt_speech_mels, speaker_embedding, finalized, offset)
                elif self.version == "cosyvoice3_streamflow":
                    res = self.generate_cv3_streamflow(speech_tokens, prompt_speech_tokens, prompt_speech_mels, speaker_embedding, finalized, offset)
                elif self.version == "cosyvoice3_streamflow_cacheprompt":
                    res = self.generate_cv3_streamflow_cacheprompt(
                        speech_tokens, prompt_speech_tokens, prompt_speech_mels, speaker_embedding, finalized, offset, speaker_id
                    )
                elif self.version == "cosyvoice3_streamflow_streaming":
                    res = self.generate_cv3_streamflow_streaming(
                        speech_tokens, prompt_speech_tokens, prompt_speech_mels, speaker_embedding, finalized, request_id
                    )
            return res
        finally:
            if finalized:
                self.cuda_cache_cleaner.empty_cache()

    def generate_cv2_cv3(
        self,
        speech_tokens: torch.Tensor,
        prompt_speech_tokens: torch.Tensor,
        prompt_speech_mels: torch.Tensor,
        speaker_embedding: torch.Tensor,
        finalized: bool,
        offset: int,
    ):
        with torch.amp.autocast("cuda", self.torch_dtype):
            speech_mels, _ = self.flow.inference(
                token=speech_tokens.to(self.device),
                token_len=torch.tensor([speech_tokens.shape[1]], dtype=torch.int32, device=self.device),
                prompt_token=prompt_speech_tokens.to(self.device),
                prompt_token_len=torch.tensor([prompt_speech_tokens.shape[1]], dtype=torch.int32, device=self.device),
                prompt_feat=prompt_speech_mels.to(self.device),
                prompt_feat_len=torch.tensor([prompt_speech_mels.shape[1]], dtype=torch.int32, device=self.device),
                embedding=speaker_embedding.to(self.device),
                streaming=not finalized,
                finalize=finalized,
            )
        speech_mels = speech_mels[:, :, offset * self.token_mel_ratio :]
        return speech_mels

    def generate_cv3_streamflow(
        self,
        speech_tokens: torch.Tensor,
        prompt_speech_tokens: torch.Tensor,
        prompt_speech_mels: torch.Tensor,
        speaker_embedding: torch.Tensor,
        finalized: bool,
        offset: int,
    ):
        with torch.amp.autocast("cuda", self.torch_dtype):
            speech_mels, *_ = self.flow.inference(
                token=torch.concat([prompt_speech_tokens, speech_tokens], dim=1).to(self.device),
                prompt_feat=prompt_speech_mels.to(self.device),
                embedding=speaker_embedding.to(self.device),
                finalize=finalized,
            )
        speech_mels = speech_mels[:, :, offset * self.token_mel_ratio :]
        return speech_mels

    def generate_cv3_streamflow_cacheprompt(
        self,
        speech_tokens: torch.Tensor,
        prompt_speech_tokens: torch.Tensor,
        prompt_speech_mels: torch.Tensor,
        speaker_embedding: torch.Tensor,
        finalized: bool,
        offset: int,
        speaker_id: str,
    ):
        this_cache = self.prompt_cache.get(speaker_id)
        if this_cache is None:
            pst = prompt_speech_tokens.to(self.device)
            psm = prompt_speech_mels.to(self.device)
            se = speaker_embedding.to(self.device)

            with torch.amp.autocast("cuda", self.torch_dtype):
                _, *caches = self.flow.inference(token=pst, prompt_feat=psm, embedding=se, finalize=False, return_cache=True)

            this_cache = {
                "cache_offset": caches[0],
                "pll_cache": caches[1],
                "dit_conv_cache": caches[2],
                "dit_attn_k_cache": caches[3],
                "dit_attn_v_cache": caches[4],
                "prompt_speech_tokens": pst[:, -self.pre_lookahead_len :],
                "prompt_speech_mels": psm,
                "speaker_embedding": se,
            }
            self.prompt_cache.put(speaker_id, this_cache)

        speech_tokens = torch.cat([this_cache["prompt_speech_tokens"], speech_tokens.to(self.device)], dim=1)
        with torch.amp.autocast("cuda", self.torch_dtype):
            speech_mels, *_ = self.flow.inference(
                token=speech_tokens,
                prompt_feat=this_cache["prompt_speech_mels"],
                embedding=this_cache["speaker_embedding"],
                finalize=finalized,
                cache_offset=this_cache["cache_offset"],
                pll_cache=this_cache["pll_cache"],
                dit_conv_cache=this_cache["dit_conv_cache"],
                dit_attn_k_cache=this_cache["dit_attn_k_cache"],
                dit_attn_v_cache=this_cache["dit_attn_v_cache"],
                return_cache=False,
            )
        speech_mels = speech_mels[:, :, offset * self.token_mel_ratio :]
        return speech_mels

    def generate_cv3_streamflow_streaming(
        self,
        speech_tokens: torch.Tensor,
        prompt_speech_tokens: torch.Tensor,
        prompt_speech_mels: torch.Tensor,
        speaker_embedding: torch.Tensor,
        finalized: bool,
        request_id: str,
    ):
        try:
            if request_id not in self.request_cache:
                self.request_cache[request_id] = {
                    "cache_offset": None,
                    "pll_cache": None,
                    "dit_conv_cache": None,
                    "dit_attn_k_cache": None,
                    "dit_attn_v_cache": None,
                }
                speech_tokens = torch.cat([prompt_speech_tokens, speech_tokens], dim=1)

            this_cache = self.request_cache[request_id]

            with torch.amp.autocast("cuda", self.torch_dtype):
                speech_mels, *caches = self.flow.inference(
                    token=speech_tokens.to(self.device),
                    prompt_feat=prompt_speech_mels.to(self.device),
                    embedding=speaker_embedding.to(self.device),
                    finalize=finalized,
                    cache_offset=this_cache["cache_offset"],
                    pll_cache=this_cache["pll_cache"],
                    dit_conv_cache=this_cache["dit_conv_cache"],
                    dit_attn_k_cache=this_cache["dit_attn_k_cache"],
                    dit_attn_v_cache=this_cache["dit_attn_v_cache"],
                    return_cache=True,
                )

            this_cache["cache_offset"] = caches[0]
            this_cache["pll_cache"] = caches[1]
            this_cache["dit_conv_cache"] = caches[2]
            if this_cache["dit_attn_k_cache"] is None:
                this_cache["dit_attn_k_cache"] = caches[3]
                this_cache["dit_attn_v_cache"] = caches[4]
            else:
                this_cache["dit_attn_k_cache"] = torch.concat((this_cache["dit_attn_k_cache"], caches[3]), dim=-2)
                this_cache["dit_attn_v_cache"] = torch.concat((this_cache["dit_attn_v_cache"], caches[4]), dim=-2)

            return speech_mels
        finally:
            if finalized:
                del self.request_cache[request_id]

    def create_model_cv2(self):
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

    def create_model_cv3(self):
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

    def create_model_cv3_streamflow(self):
        from .cv3_streamflow.model import CausalMaskedDiffWithDiT, PreLookaheadLayer, CausalConditionalCFM, DiT

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

    def convert_cv2_cv3_to_trt(self, flow_model):
        import tensorrt as trt
        from .utils import get_flow_decoder_estimator_input_shapes, convert_onnx_to_trt, set_flow_decoder_estimator, simplify_onnx, slim_onnx

        prefix = f"{self.model_dir}/flow.decoder.estimator"
        trt_path = f"{prefix}.{self.version}.{self.dtype}.plan"
        onnx_path = f"{prefix}.fp32.onnx"

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

            input_shapes = get_flow_decoder_estimator_input_shapes()

            convert_onnx_to_trt(
                onnx_path,
                trt_path,
                input_shapes,
                self.dtype,
                workspace_size=self.trt_workspace_size,
                optimization_level=3,
                timing_cache_path=os.path.join(self.compilation_cache_dir, "trt_timing.cache"),
                set_layer_precision=set_layer_precision,
            )

        with open(trt_path, "rb") as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        if estimator_engine is None:
            raise ValueError(f"Failed to load trt {trt_path}.")

        set_flow_decoder_estimator(flow_model, estimator_engine, self.device, self.num_flow_estimators)

    def build(self):
        ### create flow model
        if self.version == "cosyvoice2":
            flow = self.create_model_cv2()
        elif self.version == "cosyvoice3":
            flow = self.create_model_cv3()
        elif self.version.startswith("cosyvoice3_streamflow"):
            flow = self.create_model_cv3_streamflow()
            if "cacheprompt" in self.version:
                self.prompt_cache = PromptCacheDict(self.prompt_cache_size)
            if "streaming" in self.version:
                self.request_cache = {}
        else:
            raise ValueError(f"No `{self.version}` flow model.")

        self.token_mel_ratio = flow.token_mel_ratio
        self.pre_lookahead_len = flow.pre_lookahead_len

        ### load weights
        sd = torch.load(f"{self.model_dir}/flow.pt", map_location="cpu", weights_only=True)
        flow.load_state_dict(sd, strict=False)

        ### convert model dtype
        if self.dtype == "fp16":
            flow.half()
        elif self.dtype == "bf16":
            flow.bfloat16()

        ### move to cuda and eval mode
        flow.to(self.device).eval()

        ### use jit to accelerate flow encoder
        if self.use_jit:
            jit_encoder_path = f"{self.model_dir}/flow.encoder.{self.dtype}.zip"
            jit_encoder = torch.jit.load(jit_encoder_path, map_location=self.device)
            flow.encoder = jit_encoder

        ### use trt to accelerate flow decoder estimator
        if self.use_trt:
            if self.version.startswith("cosyvoice3_streamflow"):
                from .cv3_streamflow.convert import convert_cv3_streamflow_to_trt

                convert_cv3_streamflow_to_trt(flow, self.version, self.model_dir, self.dtype, self.device, self.trt_workspace_size)
            else:
                self.convert_cv2_cv3_to_trt(flow)

        ### compile flow model
        if self.do_compile:
            flow.inference = torch.compile(flow.inference, dynamic=True, mode="max-autotune-no-cudagraphs")

        return flow


class FlowWrapper:
    def __init__(self, flow_actor: DeploymentHandle, version: str, pre_lookahead_len: int, token_frame_rate: int):
        self.flow_actor = flow_actor
        self.pre_lookahead_len = pre_lookahead_len
        self.version = version
        self.hop_len = token_frame_rate * 2

    async def generate_nonstream(
        self,
        input_generator: AsyncGenerator[int, None],
        output_queue: asyncio.Queue,
        prompt: Prompt,
        params: Params,
    ):
        prompt_speech_tokens = prompt.flow_speech_tokens.detach()
        prompt_speech_mels = prompt.speech_mels.detach()
        speaker_embedding = prompt.speaker_embedding.detach()

        request_id, speaker_id, stream = params.id, prompt.speaker_id, params.stream
        hop_len = self.hop_len
        first_hop_padding = int(np.ceil(prompt_speech_tokens.shape[1] / hop_len) * hop_len - prompt_speech_tokens.shape[1])
        pre_lookahead_len = self.pre_lookahead_len
        flow_window_size = max(params.flow_window_size, params.flow_window_shift)
        flow_window_shift = params.flow_window_shift

        flow_handle = self.flow_actor.options(method_name="generate")

        prompt_speech_tokens_ref = ray.put(prompt_speech_tokens)
        prompt_speech_mels_ref = ray.put(prompt_speech_mels)
        speaker_embedding_ref = ray.put(speaker_embedding)

        offset = 0
        received = []
        async for token in input_generator:
            received.append(token)

            if stream:
                this_hop_len = hop_len + first_hop_padding if offset == 0 else hop_len

                if flow_window_shift > 0:
                    while offset > flow_window_size:
                        offset -= flow_window_shift
                        received = received[flow_window_shift:]

                flow_len = offset + this_hop_len + pre_lookahead_len
                if len(received) > flow_len:  # no `=` ensure remaning
                    input_tokens = ray.put(torch.tensor(received[:flow_len]).unsqueeze(0))
                    future = flow_handle.remote(
                        input_tokens,
                        prompt_speech_tokens_ref,
                        prompt_speech_mels_ref,
                        speaker_embedding_ref,
                        False,
                        offset,
                        speaker_id,
                        request_id,
                    )
                    await output_queue.put((False, future))
                    offset += this_hop_len

        if len(received) > 0:
            input_tokens = ray.put(torch.tensor(received[:]).unsqueeze(0))
            future = flow_handle.remote(
                input_tokens,
                prompt_speech_tokens_ref,
                prompt_speech_mels_ref,
                speaker_embedding_ref,
                True,
                offset,
                speaker_id,
                request_id,
            )
            await output_queue.put((True, future))

    async def generate_stream(
        self,
        input_generator: AsyncGenerator[int, None],
        output_queue: asyncio.Queue,
        prompt: Prompt,
        params: Params,
    ):
        request_id, speaker_id, stream = params.id, prompt.speaker_id, params.stream
        flow_len = self.hop_len + self.pre_lookahead_len

        flow_handle = self.flow_actor.options(method_name="generate", session_id=request_id)

        prompt_speech_tokens_ref = ray.put(prompt.flow_speech_tokens.detach())
        prompt_speech_mels_ref = ray.put(prompt.speech_mels.detach())
        speaker_embedding_ref = ray.put(prompt.speaker_embedding.detach())

        received = []
        async for token in input_generator:
            received.append(token)

            if stream and len(received) > flow_len:
                input_tokens = torch.tensor(received[:flow_len]).unsqueeze(0)
                future = flow_handle.remote(
                    input_tokens,
                    prompt_speech_tokens_ref,
                    prompt_speech_mels_ref,
                    speaker_embedding_ref,
                    False,
                    None,
                    speaker_id,
                    request_id,
                )
                await output_queue.put((False, future))
                received = received[self.hop_len :]

        if len(received) > 0:
            input_tokens = torch.tensor(received[:]).unsqueeze(0)
            future = flow_handle.remote(
                input_tokens,
                prompt_speech_tokens_ref,
                prompt_speech_mels_ref,
                speaker_embedding_ref,
                True,
                None,
                speaker_id,
                request_id,
            )
            await output_queue.put((True, future))

    async def generate(self, input_generator: AsyncGenerator[int, None], prompt: Prompt, params: Params):
        async with asyncio.TaskGroup() as tg:
            future_queue = asyncio.Queue()

            if self.version == "cosyvoice3_streamflow_streaming":
                tg.create_task(self.generate_stream(input_generator, future_queue, prompt, params))
            else:
                tg.create_task(self.generate_nonstream(input_generator, future_queue, prompt, params))

            while True:
                finalized, future = await future_queue.get()
                speech_mel = await future
                yield (finalized, speech_mel.cpu())
                if finalized:
                    break
