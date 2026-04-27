import os
import enum
from torch import Tensor
from typing import List
from pathlib import Path
from dataclasses import dataclass


os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


ROOT = Path(__file__).parents[2]
# TTS 版本
VERSION = os.getenv("VERSION", "cosyvoice2").lower()
# TTS 模型路径
TTS_MODEL_DIR = str(os.getenv("TTS_MODEL_DIR", ROOT / "assets" / "CosyVoice2-0.5B"))
# 编译缓存路径
COMPILATION_CACHE_DIR = str(os.getenv("COMPILATION_CACHE_DIR", ROOT / "assets" / "compilation_cache"))
# 文本预处理前端模型
FRONTEND_MODE = os.getenv("FRONTEND_MODE", "wetext")
# ttsfrd 资源路径
TTSFRD_RESOURCE_PATH = str(os.getenv("TTSFRD_RESOURCE_PATH", ROOT / "assets" / "CosyVoice-ttsfrd" / "resource"))
# NovaSR 模型文件路径，用于音频超分
NOVASR_MODEL_PATH = str(os.getenv("NOVASR_MODEL_PATH", ROOT / "assets" / "novasr_v2.bin"))


# 等待超时时间
WAITING_TIMEOUT = float(os.getenv("WAITING_TIMEOUT", 120.0))
# 额外配置 ray 的可用显卡
RAY_CUDA_VISIBLE_DEVICES = os.getenv("RAY_CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "0"))


#######################
#### Flow 相关配置 ####
#######################
# 限制将 flow onnx 转换为 trt 时使用的 GPU 显存大小
ONNX2TRT_WORKSPACE_SIZE = int(os.getenv("ONNX2TRT_WORKSPACE_SIZE", 2))
# 根据 GPU 显存大小量及性能设置合适的 flow decoder estimator 数量
FLOW_ESTIMATOR_COUNT = int(os.getenv("FLOW_ESTIMATOR_COUNT", 1))
# flow model 使用的 gpu 大小
FLOW_ACTOR_NUM_GPUS = float(os.getenv("FLOW_ACTOR_NUM_GPUS", 0.15))
# flow model 的实例数量
FLOW_ACTOR_COUNT = int(os.getenv("FLOW_ACTOR_COUNT", 1))
# flow model 推理精度
FLOW_DTYPE = os.getenv("FLOW_DTYPE", "tp16").lower()
# flow model 启用 jit 加速 encoder 推理，仅 `VERSION=cosyvoice2` 支持
FLOW_JIT = bool(int(os.getenv("FLOW_JIT", "0")))
# flow model 启用 tensortrt 加速 decoder estimator 推理
FLOW_TRT = bool(int(os.getenv("FLOW_TRT", "1")))
# flow model 启用 compile 加速推理
FLOW_COMPILE = bool(int(os.getenv("FLOW_COMPILE", "1")))


#######################
#### HIFT 相关配置 ####
#######################
# hift model 使用的 gpu 大小
HIFT_ACTOR_NUM_GPUS = float(os.getenv("HIFT_ACTOR_NUM_GPUS", 0.05))
# hift model 的实例数量
HIFT_ACTOR_COUNT = int(os.getenv("HIFT_ACTOR_COUNT", 1))
# hift model 启用 compile 加速推理
HIFT_COMPILE = bool(int(os.getenv("HIFT_COMPILE", "1")))
# 最大并发数
ACTOR_MAX_CONCURRENCY = int(os.getenv("ACTOR_MAX_CONCURRENCY", 100))


######################
#### LLM 相关配置 ####
######################
# llm model 引擎模式，支持 `sglang` 和 `vllm`
LLM_ENGINE_MODE = os.getenv("LLM_ENGINE_MODE", "sglang")
LLM_MAX_NUM_SILENT_TOKENS = int(os.getenv("LLM_MAX_NUM_SILENT_TOKENS", "5"))


class CosyVoiceInputType(enum.Enum):
    SINGLE = enum.auto()
    GENERATOR = enum.auto()
    QUEUE = enum.auto()


@dataclass
class Prompt:
    speaker_id: str
    speaker_embedding: Tensor
    speech_mels: Tensor
    flow_speech_tokens: Tensor
    llm_speech_tokens: List[int]
    text_tokens: List[int]
    prefix_tokens: List[int]
    suffix_tokens: List[int]


@dataclass
class Params:
    stream: bool
    speed: float = 1.0
    flow_window_size: int = 500
    flow_window_shift: int = 50
    llm_keep_orig_prompt: bool = True
    llm_min_cached_count: int = 1
    llm_min_text_cached_length: int = 1
    llm_min_speech_cached_length: int = 1
    llm_max_cached_length: int = 512
