import os
import shutil
from loguru import logger
from typing import List, AsyncGenerator
import asyncio

from ..common import VERSION, LLM_ENGINE_MODE, Prompt, Params, LLM_MAX_NUM_SILENT_TOKENS


if LLM_ENGINE_MODE == "sglang":
    from .sglang_adaption.engine import get_generation_fn
elif LLM_ENGINE_MODE == "vllm":
    from .vllm_adaption.engine import get_generation_fn


def cache_input(generator_or_list: AsyncGenerator | List[int]):
    if isinstance(generator_or_list, list):
        return generator_or_list, generator_or_list

    generator = generator_or_list
    cached = []

    async def cached_generator():
        async for i in generator:
            yield i
            cached.append(i)

    return cached_generator(), cached


class LLMWrapper:
    def __init__(self, model_dir: str, mix_ratio: List[int] = [5, 15]):
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            qwen2_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen2_config")
            shutil.copytree(qwen2_config, model_dir, dirs_exist_ok=True)

        self.mix_ratio = mix_ratio
        self.model_name = "cosyvoice3llm" if VERSION == "cosyvoice3" else "cosyvoice2llm"
        self.generate = get_generation_fn(model_dir, model_name=self.model_name, force_registration=True)

        if self.model_name == "cosyvoice2llm":
            self.num_text_tokens = 151936  # Qwen2 vocab size
            self.num_speech_tokens = 6561
            self.num_stop_tokens = 3
            self.sos_token_id = self.num_speech_tokens + self.num_stop_tokens + self.num_text_tokens
            self.task_token_id = self.sos_token_id + 1
            self.stop_token_ids = [self.num_speech_tokens + i for i in range(self.num_stop_tokens)]
            self.text_token_offset = self.num_speech_tokens + self.num_stop_tokens
            self.silent_tokens = []
        else:
            self.num_text_tokens = 151936
            self.num_speech_tokens = 6561
            self.num_stop_tokens = 200
            self.sos_token_id = self.num_speech_tokens + 0
            self.eos_token_id = self.num_speech_tokens + 1
            self.task_token_id = self.num_speech_tokens + 2
            self.fill_token_id = self.num_speech_tokens + 3
            self.stop_token_ids = [self.num_speech_tokens + i for i in range(self.num_stop_tokens)]
            self.text_token_offset = self.num_speech_tokens + self.num_stop_tokens
            self.silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]

        self.max_num_silent_tokens = LLM_MAX_NUM_SILENT_TOKENS

        logger.info("LLM is ready.")

    async def inference_unistream(
        self,
        text_tokens: List[int],
        prompt_text_tokens: List[int],
        prompt_speech_tokens: List[int],
        max_tokens_ratio=20,
        min_tokens_ratio=2,
    ) -> AsyncGenerator[int, None]:
        """Only support streaming output."""
        # offset text tokens by `text_token_offset` to distinguish speech tokens
        text_tokens = [i + self.text_token_offset for i in text_tokens]
        prompt_text_tokens = [i + self.text_token_offset for i in prompt_text_tokens]

        input_token_ids = [self.sos_token_id] + prompt_text_tokens + text_tokens + [self.task_token_id] + prompt_speech_tokens

        text_tokens_len = len(text_tokens)
        max_len = int(text_tokens_len * max_tokens_ratio)
        min_len = int(text_tokens_len * min_tokens_ratio)

        async for output_ids in self.generate(
            input_token_ids,
            stop_token_ids=self.stop_token_ids,
            max_tokens=max_len,
            min_tokens=min_len,
        ):
            if output_ids[-1] in self.stop_token_ids:
                need_yield_ids = output_ids[:-1]
            else:
                need_yield_ids = output_ids
            for token_id in need_yield_ids:
                yield token_id

    async def inference_bistream(
        self,
        text_tokens: AsyncGenerator[List[int], None],
        prompt_text_tokens: List[int],
        prompt_speech_tokens: List[int],
    ) -> AsyncGenerator[int, None]:
        """Support streaming input and streaming output."""
        text_tokens_not_input = [i + self.text_token_offset for i in prompt_text_tokens]
        speech_tokens_not_input = prompt_speech_tokens
        last_output_tokens = []
        input_token_ids = [self.sos_token_id]
        async for this_text in text_tokens:
            this_text = [i + self.text_token_offset for i in this_text]
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
                    speech_tokens_not_input = speech_tokens_not_input[self.mix_ratio[1] :]
                else:
                    break

            # inference after all speech tokens are concatenated to input
            if len(speech_tokens_not_input) == 0:
                # concatenate left text tokens to input
                if (len(last_output_tokens) > 0 and last_output_tokens[-1] == 6563) or len(input_token_ids) == 1:
                    if len(text_tokens_not_input) >= self.mix_ratio[0]:
                        input_token_ids += text_tokens_not_input[: self.mix_ratio[0]]
                        text_tokens_not_input = text_tokens_not_input[self.mix_ratio[0] :]
                    else:
                        continue

                async for output_ids in self.generate(input_token_ids, stop_token_ids=[6563]):
                    last_output_tokens = output_ids
                    if last_output_tokens[-1] == 6563:
                        need_yield_ids = last_output_tokens[:-1]
                    else:
                        need_yield_ids = last_output_tokens
                    for token_id in need_yield_ids:
                        yield token_id
                    # concatenate output to input for the next inference
                    input_token_ids.extend(need_yield_ids)

        # handle all left text tokens
        input_token_ids += text_tokens_not_input + [self.task_token_id]
        async for output_ids in self.generate(input_token_ids, stop_token_ids=[6561]):
            if output_ids[-1] == 6561:
                need_yield_ids = output_ids[:-1]
            else:
                need_yield_ids = output_ids
            for token_id in need_yield_ids:
                yield token_id

    def inference(self, text_tokens, *args):
        if isinstance(text_tokens, AsyncGenerator):
            return self.inference_bistream(text_tokens, *args)
        return self.inference_unistream(text_tokens, *args)

    async def run(self, input_generator: AsyncGenerator, output_queue: asyncio.Queue, prompt: Prompt, params: Params):
        keep_original_prompt = params.llm_keep_orig_prompt
        min_cached_count = params.llm_min_cached_count
        max_cached_length = params.llm_max_cached_length

        cached_ptt_lens, cached_ptt = [], []
        cached_pst_lens, cached_pst = [], []
        curr_generated = []
        curr_num_silent_tokens = 0

        try:
            async for input_text_tokens in input_generator:
                if len(cached_ptt) > 0:
                    if keep_original_prompt:
                        ptt_input = prompt.text_tokens + cached_ptt
                        pst_input = prompt.llm_speech_tokens + cached_pst
                    else:
                        ptt_input = cached_ptt
                        pst_input = cached_pst
                else:
                    ptt_input = prompt.text_tokens
                    pst_input = prompt.llm_speech_tokens

                # wrapped by special tokens
                ptt_input = prompt.prefix_tokens + ptt_input + prompt.suffix_tokens

                # prepare for post-processing
                curr_generated.clear()
                tt_input, cached_input = cache_input(input_text_tokens)

                generator = self.inference(tt_input, ptt_input, pst_input)
                async for speech_token in generator:
                    # keep all generated tokens to promise continuity
                    curr_generated.append(speech_token)
                    if speech_token in self.silent_tokens:
                        curr_num_silent_tokens += 1
                        if curr_num_silent_tokens >= self.max_num_silent_tokens:
                            continue
                    else:
                        curr_num_silent_tokens = 0
                    await output_queue.put(speech_token)

                cached_ptt_lens.append(len(cached_input))
                cached_pst_lens.append(len(curr_generated))
                cached_ptt += cached_input
                cached_pst += curr_generated

                while (
                    len(cached_ptt_lens) > min_cached_count and (sum(cached_ptt_lens) + sum(cached_pst_lens)) > max_cached_length
                ):
                    dropped_ptt_len = cached_ptt_lens.pop(0)
                    dropped_pst_len = cached_pst_lens.pop(0)
                    cached_ptt = cached_ptt[dropped_ptt_len:]
                    cached_pst = cached_pst[dropped_pst_len:]
        finally:
            await output_queue.put(None)
