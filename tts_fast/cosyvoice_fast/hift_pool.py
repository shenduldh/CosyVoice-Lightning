import ray
from loguru import logger
from ray.util import ActorPool
from torch.nn import functional as F
import torch
import traceback
import asyncio
import uuid
import numpy as np
import gc

from .common import (
    Params,
    VERSION,
    HIFT_ACTOR_NUM_GPUS,
    HIFT_ACTOR_COUNT,
    HIFT_COMPILE,
    ACTOR_MAX_CONCURRENCY,
)


def fade_in_out(fade_in_mel, fade_out_mel, window):
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
        + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel


class HiftActor:
    def __init__(self, model_dir, version, do_compile):
        self.model_dir = model_dir
        self.version = version
        self.do_compile = do_compile
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.hift_model = self.build()

    def ready(self):
        return True

    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()

    @ray.method(enable_task_events=False, num_returns=3)
    @torch.inference_mode()
    def generate(self, ret_flags, *infer_args):
        try:
            if self.version == "cosyvoice3":
                res = self.generate_cosyvoice3(*infer_args)
            else:
                res = self.generate_cosyvoice2(*infer_args)
            return ret_flags, res, None
        except BaseException:
            return ret_flags, None, traceback.format_exc()

    def generate_cosyvoice2(self, speech_mel: torch.Tensor, cache_source: torch.Tensor):
        speech_pcm, new_source = self.hift_model.inference(
            speech_feat=speech_mel.to(self.device),
            cache_source=cache_source.to(self.device),
        )
        speech_pcm = speech_pcm.cpu().share_memory_()
        new_source = new_source.cpu().share_memory_()
        return speech_pcm, new_source

    def generate_cosyvoice3(self, speech_mel: torch.Tensor, finalized: bool):
        speech_pcm, _ = self.hift_model.inference(speech_feat=speech_mel.to(self.device), finalize=finalized)
        speech_pcm = speech_pcm.cpu().share_memory_()
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
        from cosyvoice.hifigan.generator import CausalHiFTGenerator
        from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor

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
        ### create model
        if self.version == "cosyvoice3":
            hift = self.create_model_cosyvoice3()
        else:
            hift = self.create_model_cosyvoice2()

        ### load weight
        sd = torch.load(f"{self.model_dir}/hift.pt", weights_only=True, map_location=self.device)
        sd = {k.replace("generator.", ""): v for k, v in sd.items()}
        hift.load_state_dict(sd, strict=True)
        hift.to(self.device).eval()

        ### compile
        if self.do_compile:
            hift = torch.compile(hift, fullgraph=True, dynamic=True, mode="max-autotune")

        return hift


class HiftPool:
    def __init__(self, model_dir):
        HiftActor_class = ray.remote(HiftActor).options(num_gpus=HIFT_ACTOR_NUM_GPUS, max_concurrency=ACTOR_MAX_CONCURRENCY)
        actors = [HiftActor_class.remote(model_dir, VERSION, HIFT_COMPILE) for _ in range(HIFT_ACTOR_COUNT)]
        if all(ray.get([a.ready.remote() for a in actors])):
            logger.info(f"{self.__class__.__name__} is ready.")
        self.actors = actors
        self.pool = ActorPool(actors)

        if VERSION == "cosyvoice3":
            self.run = self.run_cosyvoice3
        else:
            self.mel_cache_len = 8
            self.source_cache_len = int(self.mel_cache_len * 480)
            self.fade_window = np.hamming(2 * self.source_cache_len)
            self.pcm_cache_len = self.source_cache_len
            self.run = self.run_cosyvoice2

        self.buffer: dict[str, asyncio.Queue] = {}
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

    async def handle_input_cosyvoice3(self, id, input_queue: asyncio.Queue, speed):
        index = 0
        received = torch.empty(1, 80, 0)
        while True:
            input = await input_queue.get()
            assert input is not None, "running is early stopped."
            finalized, speech_mel = input
            received = torch.cat([received, speech_mel], dim=2)

            if index == 0 and finalized and speed != 1.0:
                input_mel = F.interpolate(received, size=int(received.shape[2] / speed), mode="linear")
            else:
                input_mel = received

            self.submit((id, index, finalized), input_mel, finalized)
            index += 1

    async def run_cosyvoice3(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue, params: Params):
        id = uuid.uuid4().hex
        self.buffer[id] = asyncio.PriorityQueue()

        handle_input_task = asyncio.create_task(self.handle_input_cosyvoice3(id, input_queue, params.speed))

        try:
            expected_index = 0
            offset = 0
            while True:
                index, finalized, res, error = await self.buffer[id].get()
                if error:
                    raise Exception(error)
                speech_pcm = res
                if index == expected_index:
                    this_speech_pcm = speech_pcm[:, offset:].numpy().flatten()
                    await output_queue.put(this_speech_pcm)
                    offset = speech_pcm.shape[1]
                    if finalized:
                        break
                    expected_index += 1
                else:
                    await self.buffer[id].put((index, finalized, speech_pcm, None))
                    await asyncio.sleep(0)
        finally:
            self.clean()
            await output_queue.put(None)
            handle_input_task.cancel()
            del self.buffer[id]

    async def run_cosyvoice2(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue, params: Params):
        id = uuid.uuid4().hex
        self.buffer[id] = asyncio.Queue()

        index = 0
        cache_mel = torch.empty(1, 80, 0)
        cache_source = torch.zeros(1, 1, 0)
        cache_pcm = None

        try:
            while True:
                input = await input_queue.get()
                assert input is not None, "running is early stopped."
                finalized, speech_mel = input
                # concat last cached mel
                this_mel = torch.cat([cache_mel, speech_mel], dim=2)

                if index == 0 and finalized and params.speed != 1.0:
                    # adjust speaking speed
                    this_mel = F.interpolate(this_mel, size=int(this_mel.shape[2] / params.speed), mode="linear")

                self.submit((id, index, finalized), this_mel, cache_source)

                cache_mel = speech_mel[:, :, -self.mel_cache_len :]  # cache current mel

                # get result
                ret_idx, finalized, res, error = await self.buffer[id].get()
                if error:
                    raise Exception(error)
                speech_pcm, speech_source = res

                assert index == ret_idx, "return sequence is wrong."

                if cache_pcm is not None:
                    speech_pcm = fade_in_out(speech_pcm, cache_pcm, self.fade_window)

                if not finalized:
                    # remove tail pcm cached to fading in out
                    await output_queue.put(speech_pcm[:, : -self.pcm_cache_len].numpy().flatten())
                else:
                    await output_queue.put(speech_pcm.numpy().flatten())
                    break

                cache_source = speech_source[:, :, -self.source_cache_len :]  # cache speech source
                cache_pcm = speech_pcm[:, -self.pcm_cache_len :]  # cache speech pcm
                index += 1
        finally:
            self.clean()
            await output_queue.put(None)
            del self.buffer[id]
