import os
from ...common import COMPILATION_CACHE_DIR


os.environ["SGLANG_ENABLE_TORCH_INFERENCE_MODE"] = "true"
os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = "true"
os.environ["SGL_DG_USE_NVRTC"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = COMPILATION_CACHE_DIR


ENGINE_ARGS = {
    "mem_fraction_static": float(os.getenv("SGLANG_MEM_FRACTION_STATIC", 0.9)),
    "max_running_requests": int(os.getenv("SGLANG_MAX_RUNNING_REQUESTS", 64)),
    "context_length": int(os.getenv("SGLANG_CONTEXT_LENGTH", 8192)),
    "chunked_prefill_size": int(os.getenv("SGLANG_CHUNKED_PREFILL_SIZE", 8192)),
    "max_prefill_tokens": int(os.getenv("SGLANG_MAX_PREFILL_TOKENS", 131072)),
    "max_total_tokens": int(os.getenv("SGLANG_MAX_TOTAL_TOKENS", 131072)),
    ##########
    "dtype": "bfloat16",
    "attention_backend": "fa3",  # "flashinfer", "fa3", "triton"
    "load_format": "pt",
    "skip_tokenizer_init": True,
    "enable_torch_compile": True,
    "disable_cuda_graph": False,
    "enable_mixed_chunk": True,
    "sampling_backend": "flashinfer",
    "schedule_policy": "lpm",  # "lpm", "fcfs"
    "schedule_conservativeness": 1,
    "stream_interval": 1,
}


SAMPLING_PARAMS = {
    "temperature": 1,  # 不能低于 0.8，否则无法生成正常语音
    "top_p": 1,  # 不能低于 0.8，否则无法生成正常语音
    "top_k": 25,
    "min_p": 0,
    "ignore_eos": False,
    "no_stop_trim": True,
}
