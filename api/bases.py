from pydantic import BaseModel, Field
from typing import List, Union


class RemoveSpeakersInput(BaseModel):
    prompt_ids: List[str] = Field(description="要移除的参考音频 ID")


class SaveCacheInput(BaseModel):
    cache_dir: str = Field(description="缓存目录路径", default="../assets")
    filename: Union[str, None] = Field(description="缓存文件名", default=None)
    prompt_ids: List[str] = Field(description="待保存音色 ID")


class LoadCacheInput(BaseModel):
    cache_path: str = Field(description="缓存路径")
    prompt_ids: List[str] = Field(description="待加载音色 ID")


class CloneInput(BaseModel):
    prompt_audio: str = Field(description="参考音频链接")
    prompt_text: str = Field(description="参考音频文本")
    prompt_id: Union[str, None] = Field(description="参考音频 ID", default=None)
    loudness: float = Field(description="参考音频音量", default=20.0)
    audio_format: str = Field(description="参考音频格式", default="wav")
    sample_rate: int = Field(description="参考音频采样率", default=16000)


class CloneOutput(BaseModel):
    existed: bool = Field(description="音色")
    prompt_id: str = Field(description="参考音频 ID")


class TTSInput(BaseModel):
    text: str = Field(description="生成语音的文本")
    prompt_id: str = Field(description="参考音频 ID")
    audio_format: str = Field(description="合成音频格式", default="wav")
    sample_rate: int = Field(description="合成音频采样率", default=24000)
    instruct_text: Union[str | None] = Field(description="指令文本", default=None)
    return_base64: bool = Field(description="返回 base64", default=True)


class TTSOutput(BaseModel):
    audio: str = Field(description="合成音频的 base64 数据")


class TTSStreamRequestParameters(BaseModel):
    prompt_id: str = Field(description="参考音频 ID")
    audio_format: str = Field(description="合成音频格式", default="wav")
    sample_rate: int = Field(description="合成音频采样率", default=24000)
    instruct_text: Union[str | None] = Field(description="指令文本", default=None)


class TTSStreamRequestInput(BaseModel):
    req_params: TTSStreamRequestParameters = Field(description="流式 TTS 请求参数")


class TTSStreamTextInput(BaseModel):
    text: str = Field(description="需要生成语音的文本片段")
    done: bool = Field(description="文本流是否结束")


class TTSStreamOutput(BaseModel):
    id: str = Field(description="任务 ID")
    error: bool = Field(description="是否发生错误", default=False)
    is_end: bool = Field(description="合成流是否结束", default=False)
    index: int = Field(description="输出流索引", default=0)
    data: Union[str | None] = Field(
        description="流式合成音频的 base64 数据", default=None
    )
    audio_format: Union[str | None] = Field(description="合成音频格式", default=None)
    sample_rate: Union[int | None] = Field(description="合成音频采样率", default=None)
