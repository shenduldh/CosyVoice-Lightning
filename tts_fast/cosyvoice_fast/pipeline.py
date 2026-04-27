from typing import AsyncGenerator
from ruamel import yaml
import ray
import asyncio
import traceback
from loguru import logger

import tts_fast, cosyvoice, matcha, stepaudio
from .llm.wrapper import LLMWrapper
from .flow_pool import FlowPool, FLOW_ACTOR_COUNT
from .hift_pool import HiftPool, HIFT_ACTOR_COUNT
from .common import VERSION, Prompt, Params, WAITING_TIMEOUT, RAY_CUDA_VISIBLE_DEVICES


class CosyVoicePipeline:
    def __init__(self, model_dir):
        ray.init(
            num_cpus=FLOW_ACTOR_COUNT + HIFT_ACTOR_COUNT,
            runtime_env={
                "env_vars": {"CUDA_VISIBLE_DEVICES": RAY_CUDA_VISIBLE_DEVICES},
                "py_modules": [tts_fast, cosyvoice, matcha, stepaudio],
                "excludes": ["__pycache__"],
            },
        )

        self.set_config(model_dir)

        self.llm = LLMWrapper(model_dir)
        self.flow_pool = FlowPool(model_dir, self.pre_lookahead_len, self.token_frame_rate)
        self.hift_pool = HiftPool(model_dir)

    def set_config(self, model_dir):
        config_name = "cosyvoice3" if VERSION == "cosyvoice3" else "cosyvoice2"
        with open(f"{model_dir}/{config_name}.yaml", "r") as f:
            cfg = yaml.YAML().load(f)
            self.token_mel_ratio: int = cfg["token_mel_ratio"]
            self.pre_lookahead_len: int = cfg["flow"]["pre_lookahead_len"]
            self.token_frame_rate: int = cfg["token_frame_rate"]
            self.sample_rate: int = cfg["sample_rate"]

    async def generate(self, input_generator: AsyncGenerator, prompt: Prompt, params: Params):
        llm2flow_queue = asyncio.Queue()
        flow2hift_queue = asyncio.Queue()
        output_queue = asyncio.Queue()

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.llm.run(input_generator, llm2flow_queue, prompt, params))
                tg.create_task(self.flow_pool.run(llm2flow_queue, flow2hift_queue, prompt, params))
                tg.create_task(self.hift_pool.run(flow2hift_queue, output_queue, params))

                while True:
                    async with asyncio.timeout(WAITING_TIMEOUT):
                        res = await output_queue.get()
                    if res is None:
                        break
                    yield res
        except* BaseException:
            logger.error(f"Error in {self.__class__.__name__}:\n{''.join(traceback.format_exc())}")
            raise ValueError("Generation failed.") from None
        finally:
            del llm2flow_queue
            del flow2hift_queue
            del output_queue
