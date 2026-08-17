import threading
from collections.abc import AsyncGenerator
from ray import serve
from ray.serve.handle import DeploymentHandle
from torch.nn import functional as F
import torch
import traceback
import asyncio
import numpy as np

from .common import Params
from .utils import CudaCacheCleaner


def fade_in_out(fade_in_mel, fade_out_mel, window):
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel


@serve.deployment
class HiftActor:
    def __init__(
        self,
        model_dir,
        version,
        use_trt,
        do_compile,
        use_cuda_cache_cleaner,
        cuda_cache_clean_delay,
        cuda_cache_clean_interval,
    ):
        self.model_dir = model_dir
        self.version = version
        self.use_trt = use_trt
        self.do_compile = do_compile
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.hift_model = self.build()
        self.lock = threading.Lock()
        self.cuda_cache_cleaner = CudaCacheCleaner(
            cuda_cache_clean_delay, cuda_cache_clean_interval, device=self.device, enabled=use_cuda_cache_cleaner
        )

    def generate(self, speech_mel, cache_source, finalized):
        try:
            with self.lock, self.cuda_cache_cleaner.track():
                if self.version.startswith("cosyvoice3"):
                    res = self.generate_cosyvoice3(speech_mel, finalized)
                else:
                    res = self.generate_cosyvoice2(speech_mel, cache_source)
            return res
        finally:
            if finalized:
                self.cuda_cache_cleaner.empty_cache()

    def generate_cosyvoice2(self, speech_mel: torch.Tensor, cache_source: torch.Tensor):
        speech_pcm, new_source = self.hift_model.inference(
            speech_feat=speech_mel.to(self.device),
            cache_source=cache_source.to(self.device),
        )
        speech_pcm = speech_pcm.cpu()
        new_source = new_source.cpu()
        return speech_pcm, new_source

    def generate_cosyvoice3(self, speech_mel: torch.Tensor, finalized: bool):
        speech_pcm = self.hift_model.inference(speech_feat=speech_mel.to(self.device), finalize=finalized)
        speech_pcm = speech_pcm.cpu()
        return speech_pcm

    def create_model_cosyvoice2(self):
        from cosyvoice.hifigan.generator import HiFTGenerator
        from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor

        return HiFTGenerator(
            in_channels=80,
            base_channels=512,
            nb_harmonics=8,
            sampling_rate=24000,
            nsf_alpha=0.1,
            nsf_sigma=0.003,
            nsf_voiced_threshold=10,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            istft_params={"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            lrelu_slope=0.1,
            audio_limit=0.99,
            f0_predictor=ConvRNNF0Predictor(
                num_class=1,
                in_channels=80,
                cond_channels=512,
            ),
        )

    def create_model_cosyvoice3(self):
        from .cv3_hift.model import CausalHiFTGenerator, CausalConvRNNF0Predictor

        return CausalHiFTGenerator(
            in_channels=80,
            base_channels=512,
            nb_harmonics=8,
            sampling_rate=24000,
            nsf_alpha=0.1,
            nsf_sigma=0.003,
            nsf_voiced_threshold=10,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            istft_params={"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            lrelu_slope=0.1,
            audio_limit=0.99,
            conv_pre_look_right=4,
            f0_predictor=CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512),
        )

    def build(self):
        import torch

        ### create model
        if self.version.startswith("cosyvoice3"):
            hift = self.create_model_cosyvoice3()
        else:
            hift = self.create_model_cosyvoice2()

        ### load weight
        sd = torch.load(f"{self.model_dir}/hift.pt", weights_only=True, map_location=self.device)
        sd = {k.replace("generator.", ""): v for k, v in sd.items()}
        hift.load_state_dict(sd, strict=False)
        hift.to(self.device).eval()

        if self.version.startswith("cosyvoice3"):
            hift.remove_parametrizations()

            if self.do_compile:
                torch._dynamo.config.recompile_limit = 32
                for i in range(len(hift.source_downs)):
                    hift.source_downs[i].forward = torch.compiler.disable(hift.source_downs[i].forward)

            if self.use_trt:
                from .cv3_hift.convert import convert_hift_to_trt

                hift = convert_hift_to_trt(hift, self.model_dir)
            else:
                hift.use_high_precision()

        ### compile
        if self.do_compile:
            hift.inference = torch.compile(hift.inference, dynamic=True, mode="max-autotune-no-cudagraphs")

        return hift


class HiftWrapper:
    def __init__(
        self,
        hift_actor: DeploymentHandle,
        version: str,
    ):
        self.hift_actor = hift_actor
        self.version = version
        if not version.startswith("cosyvoice3"):
            self.mel_cache_len = 8
            self.source_cache_len = int(self.mel_cache_len * 480)
            self.fade_window = np.hamming(2 * self.source_cache_len)
            self.pcm_cache_len = self.source_cache_len

    async def generate_cv2(self, input_generator: AsyncGenerator, params: Params):
        first = True
        cache_mel = torch.empty(1, 80, 0)
        cache_source = torch.zeros(1, 1, 0)
        cache_pcm = None

        async for finalized, speech_mel in input_generator:
            this_mel = torch.cat([cache_mel, speech_mel], dim=2)

            if first and finalized and params.speed != 1.0:
                # adjust speaking speed
                this_mel = F.interpolate(this_mel, size=int(this_mel.shape[2] / params.speed), mode="linear")
                first = False

            res = await self.hift_actor.generate.remote(this_mel, cache_source, finalized)
            speech_pcm, speech_source = res

            if cache_pcm is not None:
                speech_pcm = fade_in_out(speech_pcm, cache_pcm, self.fade_window)

            if not finalized:
                # remove tail pcm cached to fading in out
                yield speech_pcm[:, : -self.pcm_cache_len].numpy().flatten()
            else:
                yield speech_pcm.numpy().flatten()

            cache_mel = speech_mel[:, :, -self.mel_cache_len :]  # cache current mel
            cache_source = speech_source[:, :, -self.source_cache_len :]  # cache speech source
            cache_pcm = speech_pcm[:, -self.pcm_cache_len :]  # cache speech pcm

    async def generate_cv3(self, input_generator: AsyncGenerator, params: Params):
        output_queue = asyncio.Queue()

        async def producer():
            first = True
            received = torch.empty(1, 80, 0)
            async for finalized, speech_mel in input_generator:
                received = torch.concat([received, speech_mel], dim=2)

                if first and finalized and params.speed != 1.0:
                    input_mel = F.interpolate(received, size=int(received.shape[2] / params.speed), mode="linear")
                    first = False
                else:
                    input_mel = received

                future = self.hift_actor.generate.remote(input_mel, None, finalized)
                await output_queue.put(future)

                if finalized:
                    await output_queue.put(None)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(producer())

            offset = 0
            while True:
                future = await output_queue.get()
                if future is None:
                    break
                speech_pcm = await future
                yield speech_pcm[:, offset:].numpy().flatten()
                offset = speech_pcm.shape[1]

    def generate(self, *args, **kwargs):
        if self.version.startswith("cosyvoice3"):
            return self.generate_cv3(*args, **kwargs)
        else:
            return self.generate_cv2(*args, **kwargs)
