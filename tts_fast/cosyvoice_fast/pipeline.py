import os
from ruamel import yaml
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve.schema import LoggingConfig
import asyncio
from datetime import datetime
import zmq

import cosyvoice, matcha
from .llm.wrapper import LLMWrapper
from .flow_wrapper import FlowWrapper, FlowActor
from .hift_wrapper import HiftWrapper, HiftActor
from .common import (
    ROOT,
    VERSION,
    Prompt,
    Params,
    TTS_MODEL_DIR,
    NUM_FLOW_ACTORS,
    NUM_HIFT_ACTORS,
    FLOW_NUM_GPUS,
    HIFT_NUM_GPUS,
    FLOW_DTYPE,
    FLOW_JIT,
    FLOW_TRT,
    FLOW_COMPILE,
    FLOW_PROMPT_CACHE_SIZE,
    HIFT_TRT,
    HIFT_COMPILE,
    LLM_NUM_GPUS,
    LLM_MAX_NUM_SILENT_TOKENS,
    LLM_ENGINE_MODE,
    FLOW_TRT_WORKSPACE_SIZE,
    COMPILATION_CACHE_DIR,
    NUM_FLOW_ESTIMATORS,
    RAY_MAX_ONGOING_REQUESTS,
    USE_CUDA_CACHE_CLEANER,
    CUDA_CACHE_CLEAN_DELAY,
    CUDA_CACHE_CLEAN_INTERVAL,
)


@serve.deployment
class CosyVoicePipeline:
    def __init__(
        self,
        llm_handle: DeploymentHandle,
        flow_handle: DeploymentHandle,
        hift_handle: DeploymentHandle,
        version: str,
        pre_lookahead_len: int,
        token_frame_rate: int,
    ):
        factory_options = {"request_serialization": "msgpack", "response_serialization": "msgpack"}
        self.llm = llm_handle.options(stream=True, **factory_options)
        self.flow = FlowWrapper(flow_handle.options(**factory_options), version, pre_lookahead_len, token_frame_rate)
        self.hift = HiftWrapper(hift_handle.options(**factory_options), version)

    async def generate(self, zmq_socket_addr: str, prompt: Prompt, params: Params):
        llm_output_generator = self.llm.run.remote(zmq_socket_addr, prompt, params)
        flow_output_generator = self.flow.generate(llm_output_generator, prompt, params)
        async for i in self.hift.generate(flow_output_generator, params):
            yield i


def get_synthesizer():
    version = VERSION

    model_dir = TTS_MODEL_DIR
    cache_dir = COMPILATION_CACHE_DIR

    num_flow_actors = NUM_FLOW_ACTORS
    flow_num_gpus = FLOW_NUM_GPUS
    flow_dtype = FLOW_DTYPE
    flow_use_jit = FLOW_JIT
    flow_use_trt = FLOW_TRT
    flow_trt_workspace_size = FLOW_TRT_WORKSPACE_SIZE
    num_flow_estimators = NUM_FLOW_ESTIMATORS
    flow_do_compile = FLOW_COMPILE
    flow_prompt_cache_size = FLOW_PROMPT_CACHE_SIZE

    num_hift_actors = NUM_HIFT_ACTORS
    hift_num_gpus = HIFT_NUM_GPUS
    hift_do_compile = HIFT_COMPILE
    hift_use_trt = HIFT_TRT

    llm_max_num_silent_tokens = LLM_MAX_NUM_SILENT_TOKENS
    llm_engine_mode = LLM_ENGINE_MODE

    num_llm_replicas = 1
    num_pipeline_replicas = 1
    ray_max_ongoing_requests = RAY_MAX_ONGOING_REQUESTS
    use_cuda_cache_cleaner = USE_CUDA_CACHE_CLEANER
    cuda_cache_clean_delay = CUDA_CACHE_CLEAN_DELAY
    cuda_cache_clean_interval = CUDA_CACHE_CLEAN_INTERVAL

    # init ray
    ray.init(
        num_cpus=num_flow_actors + num_hift_actors + num_llm_replicas + num_pipeline_replicas,
        runtime_env={"py_modules": [cosyvoice, matcha], "excludes": ["__pycache__"]},
        include_dashboard=False,
    )

    # get config
    config_name = "cosyvoice3" if version.startswith("cosyvoice3") else "cosyvoice2"
    with open(f"{model_dir}/{config_name}.yaml", "r") as f:
        cfg = yaml.YAML().load(f)
        token_mel_ratio: int = cfg["token_mel_ratio"]
        sample_rate: int = cfg["sample_rate"]
        pre_lookahead_len: int = cfg["flow"]["pre_lookahead_len"]
        token_frame_rate: int = cfg["token_frame_rate"]

    # build models
    flow_actor = FlowActor.options(
        max_ongoing_requests=ray_max_ongoing_requests,
        num_replicas=num_flow_actors,
        ray_actor_options={"num_gpus": flow_num_gpus, "num_cpus": 1},
    ).bind(
        model_dir,
        version,
        flow_use_jit,
        flow_use_trt,
        flow_dtype,
        flow_trt_workspace_size,
        num_flow_estimators,
        flow_do_compile,
        cache_dir,
        flow_prompt_cache_size,
        use_cuda_cache_cleaner,
        cuda_cache_clean_delay,
        cuda_cache_clean_interval,
    )
    hift_actor = HiftActor.options(
        max_ongoing_requests=ray_max_ongoing_requests,
        num_replicas=num_hift_actors,
        ray_actor_options={"num_gpus": hift_num_gpus, "num_cpus": 1},
    ).bind(model_dir, version, hift_use_trt, hift_do_compile, use_cuda_cache_cleaner, cuda_cache_clean_delay, cuda_cache_clean_interval)
    llm = LLMWrapper.options(
        num_replicas=num_llm_replicas,
        max_ongoing_requests=ray_max_ongoing_requests,
        ray_actor_options={"num_gpus": LLM_NUM_GPUS, "num_cpus": 1},
    ).bind(
        model_dir=model_dir,
        cache_dir=cache_dir,
        version=version,
        engine_mode=llm_engine_mode,
        max_num_silent_tokens=llm_max_num_silent_tokens,
    )
    pipeline = CosyVoicePipeline.options(
        num_replicas=num_pipeline_replicas,
        max_ongoing_requests=ray_max_ongoing_requests,
        ray_actor_options={"num_cpus": 1},
    ).bind(llm, flow_actor, hift_actor, version, pre_lookahead_len, token_frame_rate)

    # get handle
    handle = serve.run(
        pipeline,
        route_prefix=None,
        logging_config=LoggingConfig(encoding="TEXT", log_level="INFO", logs_dir=os.getenv("LOG_DIR")),
    ).options(stream=True)

    # create zmq
    temp_zmq_dir = f"/tmp/cosyvoice_api/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(temp_zmq_dir, exist_ok=True)
    zmq_ctx = zmq.asyncio.Context()

    async def generate(input_generator, prompt: Prompt, params: Params):
        zmq_file_path = os.path.join(temp_zmq_dir, f"{params.id}.ipc")
        addr = f"ipc://{zmq_file_path}"
        socket = zmq_ctx.socket(zmq.PUSH)
        pusher_task = None

        try:
            socket.bind(addr)
            output_generator = handle.generate.remote(addr, prompt, params)

            async def data_pusher():
                async for data in input_generator:
                    await socket.send_pyobj(data)
                await socket.send_pyobj(None)

            pusher_task = asyncio.create_task(data_pusher())

            async for i in output_generator:
                yield i

        finally:
            try:
                output_generator.cancel()
            except:
                pass

            try:
                if pusher_task and not pusher_task.done():
                    pusher_task.cancel()
                    await pusher_task
            except:
                pass

            try:
                socket.close(linger=0)
            except:
                pass

            try:
                if os.path.exists(zmq_file_path):
                    os.remove(zmq_file_path)
            except:
                pass

    return generate, sample_rate
