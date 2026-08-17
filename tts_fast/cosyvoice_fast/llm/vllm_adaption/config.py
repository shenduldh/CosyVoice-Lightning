import os
import logging


logging.getLogger("vllm.inputs.preprocess").setLevel(logging.ERROR)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN_VLLM_V1"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER_VLLM_V1"


from vllm.sampling_params import RequestOutputKind


ENGINE_ARGS = {
    "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.5")),
    "block_size": int(os.getenv("VLLM_BLOCK_SIZE", "16")),
    "max_num_batched_tokens": int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "8192")),  # 单次迭代最大处理 token 数
    "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
    "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "16")),  # 单次迭代并发数
    "quantization": "fp8",
    "dtype": "bfloat16",
    "swap_space": 0,
    "skip_tokenizer_init": True,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": False,
    "load_format": "pt",
    "enforce_eager": False,
    "enable_log_requests": False,
    "disable_log_stats": True,
}


SAMPLING_PARAMS = {
    "temperature": 1,  # 不能低于 0.8，否则无法生成正常语音
    "top_p": 1,  # 不能低于 0.8，否则无法生成正常语音
    "top_k": 25,
    "detokenize": False,
    "ignore_eos": False,
    "output_kind": RequestOutputKind.DELTA,
}
