import os
import shutil
from typing import List, AsyncGenerator
import torch
import uuid
import uvloop
import threading
import asyncio


ENGINE_MODE = os.getenv("LLM_ENGINE_MODE", "sglang")
if ENGINE_MODE == "sglang":
    import sglang
    from .sglang_adaption.config import ENGINE_ARGS, SAMPLING_PARAMS
    from .sglang_adaption.register import register_model
elif ENGINE_MODE == "vllm":
    import vllm
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
    from .vllm_adaption.config import ENGINE_ARGS, SAMPLING_PARAMS
    from .vllm_adaption.register import register_model


def tensor_to_list(tensor):
    return tensor.view(-1).cpu().numpy().tolist()


class CosyVoice2LLMWrapper:
    def __init__(
        self,
        model_dir,
        mix_ratio: List[int] = [5, 15],
    ):
        register_model(forced=False)
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            src_config_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "qwen2_config"
            )
            shutil.copytree(src_config_dir, model_dir, dirs_exist_ok=True)

        if ENGINE_MODE == "sglang":
            self.engine = sglang.Engine(model_path=model_dir, **ENGINE_ARGS)
            self.engine_generate_fn = self.sglang_generate
        elif ENGINE_MODE == "vllm":
            self.engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(model=model_dir, **ENGINE_ARGS)
            )
            if ".".join(map(str, vllm.__version_tuple__[:2])) in ["0.8", "0.9"]:
                setattr(
                    self.engine.processor,
                    "_validate_model_inputs",
                    lambda *args, **kwargs: None,
                )
                setattr(
                    self.engine.processor.input_preprocessor,
                    "get_eos_token_id",
                    lambda *args, **kwargs: None,
                )
            self.engine_generate_fn = self.vllm_generate

        self.mix_ratio = mix_ratio
        self.num_speech_tokens = 6561 + 3
        self.num_text_tokens = 151936  # Qwen2's vocab_size
        self.sos_eos_token_id = self.num_speech_tokens + self.num_text_tokens
        self.task_token_id = self.sos_eos_token_id + 1
        self.zero_token_id = self.task_token_id + 1
        self.stop_token_ids = [6561, 6562, 6563]

        self.loop = uvloop.new_event_loop()
        threading.Thread(
            target=lambda: (asyncio.set_event_loop(self.loop), self.loop.run_forever()),
            daemon=True,
        ).start()

    async def sglang_generate(
        self, input_token_ids, request_id, stop_token_ids, max_tokens=None, min_tokens=0
    ):
        sampling_params = {
            **SAMPLING_PARAMS,
            "stop_token_ids": stop_token_ids,
            "max_new_tokens": max_tokens,
            "min_new_tokens": 0,
        }
        generator = await self.engine.async_generate(
            input_ids=input_token_ids,
            sampling_params=sampling_params,
            stream=True,
        )
        last_completion_tokens = 0
        async for output in generator:
            yield output["output_ids"][last_completion_tokens:]
            last_completion_tokens = output["meta_info"]["completion_tokens"]

    async def vllm_generate(
        self, input_token_ids, request_id, stop_token_ids, max_tokens=None, min_tokens=0
    ):
        sampling_params = SamplingParams(
            **SAMPLING_PARAMS,
            stop_token_ids=stop_token_ids,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )
        async for output in self.engine.generate(
            {"prompt_token_ids": input_token_ids},
            sampling_params=sampling_params,
            request_id=request_id or uuid.uuid4().hex,
        ):
            yield output.outputs[0].token_ids

    async def background_generate(self, *args, q: asyncio.Queue, main_loop, **kwargs):
        try:
            async for tokens in self.engine_generate_fn(*args, **kwargs):
                main_loop.call_soon_threadsafe(q.put_nowait, tokens)
        finally:
            main_loop.call_soon_threadsafe(q.put_nowait, None)

    async def call_engine_generation(self, *args, **kwargs):
        q = asyncio.Queue()
        main_loop = asyncio.get_running_loop()
        asyncio.run_coroutine_threadsafe(self.background_generate(*args, **kwargs, q=q, main_loop=main_loop), self.loop)
        while True:
            tokens = await q.get()
            if tokens is None:
                break
            yield tokens

    async def inference(
        self,
        text: torch.Tensor,
        prompt_text_tokens: torch.Tensor,
        prompt_speech_tokens: torch.Tensor,
        request_id=None,
        max_tokens_ratio=20,
        min_tokens_ratio=2,
    ) -> AsyncGenerator[int, None]:
        """Only support streaming output."""
        # offset text tokens by `num_speech_tokens` to distinguish speech tokens
        text_tokens = tensor_to_list(text + self.num_speech_tokens)
        prompt_text_tokens = tensor_to_list(prompt_text_tokens + self.num_speech_tokens)
        prompt_speech_tokens = tensor_to_list(prompt_speech_tokens)

        input_token_ids = (
            [self.sos_eos_token_id]
            + prompt_text_tokens
            + text_tokens
            + [self.task_token_id]
            + prompt_speech_tokens
        )

        text_tokens_len = len(text_tokens)
        max_len = int(text_tokens_len * max_tokens_ratio)
        min_len = int(text_tokens_len * min_tokens_ratio)

        async for output_ids in self.call_engine_generation(
            input_token_ids,
            request_id=request_id,
            stop_token_ids=self.stop_token_ids,
            max_tokens=max_len,
            min_tokens=min_len,
        ):
            if output_ids[-1] in self.stop_token_ids:
                need_out_tokens = output_ids[:-1]
            else:
                need_out_tokens = output_ids
            for token in need_out_tokens:
                yield token

    async def inference_bistream(
        self,
        text: AsyncGenerator[torch.Tensor, None],
        prompt_text_tokens: torch.Tensor,
        prompt_speech_tokens: torch.Tensor,
        request_id=None,
    ) -> AsyncGenerator[int, None]:
        """Support streaming input and streaming output."""
        text_tokens_not_input = tensor_to_list(
            prompt_text_tokens + self.num_speech_tokens
        )
        speech_tokens_not_input = tensor_to_list(prompt_speech_tokens)
        last_output_tokens = []
        input_token_ids = [self.sos_eos_token_id]
        async for this_text in text:
            this_text = tensor_to_list(this_text + self.num_speech_tokens)
            text_tokens_not_input += this_text
            # when there are still speech tokens left
            # align them with text tokens and concatenate them to input
            while len(speech_tokens_not_input) != 0:
                if len(text_tokens_not_input) >= self.mix_ratio[0]:
                    curr_text_tokens = text_tokens_not_input[: self.mix_ratio[0]]
                    curr_speech_tokens = speech_tokens_not_input[: self.mix_ratio[1]]
                    input_token_ids += curr_text_tokens + curr_speech_tokens
                    # update tokens not input
                    text_tokens_not_input = text_tokens_not_input[self.mix_ratio[0] :]
                    speech_tokens_not_input = speech_tokens_not_input[
                        self.mix_ratio[1] :
                    ]
                else:
                    break

            # inference after all speech tokens are concatenated to input
            if len(speech_tokens_not_input) == 0:
                # concatenate left text tokens to input
                if (
                    len(last_output_tokens) > 0 and last_output_tokens[-1] == 6563
                ) or len(input_token_ids) == 1:
                    if len(text_tokens_not_input) >= self.mix_ratio[0]:
                        input_token_ids += text_tokens_not_input[: self.mix_ratio[0]]
                        text_tokens_not_input = text_tokens_not_input[
                            self.mix_ratio[0] :
                        ]
                    else:
                        continue

                async for output_ids in self.call_engine_generation(
                    input_token_ids, request_id=request_id, stop_token_ids=[6563]
                ):
                    last_output_tokens = output_ids
                    if last_output_tokens[-1] == 6563:
                        need_out_tokens = last_output_tokens[:-1]
                    else:
                        need_out_tokens = last_output_tokens
                    for token in need_out_tokens:
                        yield token
                    # concatenate output to input for the next inference
                    input_token_ids.extend(need_out_tokens)

        # handle all left text tokens
        input_token_ids += text_tokens_not_input + [self.task_token_id]
        async for output_ids in self.call_engine_generation(
            input_token_ids, request_id=request_id, stop_token_ids=[6561]
        ):
            if output_ids[-1] == 6561:
                need_out_tokens = output_ids[:-1]
            else:
                need_out_tokens = output_ids
            for token in need_out_tokens:
                yield token

    def __call__(
        self,
        text: torch.Tensor,
        prompt_text_tokens: torch.Tensor,
        prompt_speech_tokens: torch.Tensor,
        request_id=None,
    ):
        infer_args = (text, prompt_text_tokens, prompt_speech_tokens, request_id)
        if isinstance(text, AsyncGenerator):
            return self.inference_bistream(*infer_args)
        else:
            return self.inference(*infer_args)
