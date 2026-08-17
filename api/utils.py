import os
import numpy as np
from pydub import AudioSegment
import librosa
from urllib.request import urlopen
from io import BytesIO
import soundfile as sf
import io
import base64
import struct
from typing import Literal
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
import traceback
from tempfile import NamedTemporaryFile
import zhon
import string
import re
import torch
import av


def path_to_root(*paths):
    root = Path(__file__).parents[1]
    if len(paths) > 0:
        return Path(root, *paths).as_posix()
    else:
        return root.as_posix()


def truncate_long_str(obj, max_len=70, ellipsis="......"):
    if isinstance(obj, str):
        if len(obj) > max_len:
            return obj[:max_len] + ellipsis
        return obj
    elif isinstance(obj, (tuple, list)):
        return [truncate_long_str(i, max_len, ellipsis) for i in obj]
    elif isinstance(obj, dict):
        return {k: truncate_long_str(v, max_len, ellipsis) for k, v in obj.items()}
    else:
        return obj


def whats_wrong_with(e):
    return traceback.format_exception_only(e)[-1].strip()


def get_gpu_utilization():
    return {"allocated": torch.cuda.memory_allocated() / 1024**2, "reserved": torch.cuda.memory_reserved() / 1024**2}


class Stream:
    def __init__(self, capacity: int, dtype=np.float32):
        self.capacity = capacity
        self.buffer = np.zeros(self.capacity * 2, dtype=dtype)
        self.head = 0
        self.rear = 0
        self.size = 0

    def add(self, data: np.ndarray):
        data_size = len(data)
        if data_size > self.capacity:
            data = data[-self.capacity :]
            data_size = self.capacity

        end = self.rear + data_size

        # write twice to both the original position and mirrored position
        if end <= self.capacity:
            self.buffer[self.rear : end] = data
            self.buffer[self.rear + self.capacity : end + self.capacity] = data  # mirror
        else:
            pivot = self.capacity - self.rear
            self.buffer[self.rear : self.capacity] = data[:pivot]
            self.buffer[self.rear + self.capacity : self.capacity + self.capacity] = data[:pivot]  # mirror
            self.buffer[: data_size - pivot] = data[pivot:]
            self.buffer[self.capacity : self.capacity + data_size - pivot] = data[pivot:]  # mirror

        self.rear = (self.rear + data_size) % self.capacity
        new_size = self.size + data_size
        if new_size > self.capacity:
            self.head = self.rear
            self.size = self.capacity
        else:
            self.size = new_size

    def read(self, length: int):
        if self.size < length:
            return None

        data = self.buffer[self.head : self.head + length]
        self.head = (self.head + length) % self.capacity
        self.size -= length
        return data


async def async_repack(
    generator: AsyncGenerator[np.ndarray, None], min_size=16000 * 1, max_size=16000 * 10, capacity=None
) -> AsyncGenerator[np.ndarray, None]:
    capacity = max_size * 2 if capacity is None else capacity
    stream = Stream(capacity, dtype=np.float32)
    async for chunk in generator:
        stream.add(chunk)
        while stream.size >= min_size:
            output_size = min(max_size, stream.size)
            output = stream.read(output_size)
            yield output
    if stream.size > 0:
        output = stream.read(stream.size)
        yield output


def repack(generator: Generator[np.ndarray, None, None], min_size=16000 * 1, max_size=16000 * 10, capacity=None) -> Generator[np.ndarray, None, None]:
    capacity = max_size * 2 if capacity is None else capacity
    stream = Stream(capacity, dtype=np.float32)
    for chunk in generator:
        stream.add(chunk)
        while stream.size >= min_size:
            output_size = min(max_size, stream.size)
            output = stream.read(output_size)
            yield output
    if stream.size > 0:
        output = stream.read(stream.size)
        yield output


def add_wav_header(audio_bytes: bytes, sample_rate=16000, num_channels=1, bits_per_sample=16):
    if audio_bytes.startswith(b"RIFF"):
        return audio_bytes

    SIGNED_INT = struct.Struct("<i")
    UNSIGNED_SHORT = struct.Struct("<H")
    num_samples = len(audio_bytes)

    header = b"RIFF"
    header += SIGNED_INT.pack(num_samples + 36)
    header += b"WAVEfmt "
    header += b"\x10\x00\x00\x00"  # fmt chunk size
    header += b"\x01\x00"  # audio format (1 is PCM)
    header += UNSIGNED_SHORT.pack(num_channels)
    header += SIGNED_INT.pack(sample_rate)
    header += SIGNED_INT.pack(sample_rate * num_channels * bits_per_sample // 8)  # bytes per sample
    header += UNSIGNED_SHORT.pack(num_channels * bits_per_sample // 8)  # block alignment
    header += UNSIGNED_SHORT.pack(bits_per_sample)
    header += b"data"
    header += SIGNED_INT.pack(num_samples)

    return header + audio_bytes


short_info = np.iinfo(np.short)
min_short = short_info.min
max_short = short_info.max
abs_max_short: int = 2 ** (short_info.bits - 1)
offset = min_short + abs_max_short


def float32_to_int16(audio_ndarray: np.ndarray):
    return (audio_ndarray * abs_max_short + offset).clip(min_short, max_short).astype(np.short)


def int16_to_float32(audio_ndarray: np.ndarray):
    return (audio_ndarray.astype(np.float32) - offset) / abs_max_short


def to_mono(audio_ndarray: np.ndarray) -> np.ndarray:
    if len(audio_ndarray.shape) == 2:
        n1, n2 = audio_ndarray.shape
        channel_axis = 0 if n1 < n2 else 1
        return audio_ndarray.mean(axis=channel_axis)
    return audio_ndarray


def save_audio(audio_ndarray: np.ndarray, path: str, sample_rate=16000):
    audio_ndarray = float32_to_int16(audio_ndarray)
    sf.write(path, audio_ndarray, samplerate=sample_rate)


def ndarray_to_pydub(audio_ndarray: np.ndarray, sample_rate=16000, num_channels=1, sample_width=2) -> AudioSegment:
    audio_segment = AudioSegment.from_raw(
        io.BytesIO(float32_to_int16(audio_ndarray).tobytes()),
        sample_width=sample_width,
        frame_rate=sample_rate,
        channels=num_channels,
    )
    return audio_segment


def load_audio_segment(audio_path: str, format: str, sample_rate=16000):
    if audio_path.startswith("http"):
        audio = BytesIO(urlopen(audio_path).read())
    else:
        audio = audio_path

    audio_seg: AudioSegment = AudioSegment.from_file(audio, format=format)
    audio_seg = audio_seg.set_frame_rate(sample_rate).set_channels(1)
    return audio_seg


def pydub_to_ndarray(audio_seg: AudioSegment):
    audio_ndarray = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
    scale = 1.0 / float(1 << ((8 * audio_seg.sample_width) - 1))
    audio_ndarray *= scale
    return audio_ndarray


def bytes_no_header_to_ndarray(audio_bytes: bytes, sample_rate=16000):
    audio_bytes = add_wav_header(audio_bytes, sample_rate)
    audio_ndarray, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
    return audio_ndarray


def ndarray_to_bytes_with_wav_header(audio_ndarray: np.ndarray, sample_rate=16000):
    audio_bytes = float32_to_int16(to_mono(audio_ndarray)).tobytes()
    audio_bytes = add_wav_header(audio_bytes, sample_rate)
    return audio_bytes


def ndarray_to_base64_no_header(audio_ndarray: np.ndarray):
    audio_bytes = float32_to_int16(to_mono(audio_ndarray)).tobytes()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    return audio_base64


def base64_no_header_to_ndarray(audio_base64: str, sample_rate=16000):
    audio_bytes = base64.b64decode(audio_base64.encode("utf-8"))
    audio_bytes = add_wav_header(audio_bytes, sample_rate)
    audio_ndarray, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
    return audio_ndarray


def ndarray_to_base64_with_wav_header(audio_ndarray: np.ndarray, sample_rate=16000, resample_rate=16000):
    if resample_rate != sample_rate:
        audio_ndarray = librosa.resample(audio_ndarray, orig_sr=sample_rate, target_sr=resample_rate)
    audio_bytes = float32_to_int16(to_mono(audio_ndarray)).tobytes()
    audio_bytes = add_wav_header(audio_bytes, sample_rate)
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    return audio_base64


def base64_with_wav_header_to_ndarray(audio_base64: str, sample_rate=16000):
    audio_bytes = base64.b64decode(audio_base64.encode("utf-8"))
    audio_ndarray, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
    return audio_ndarray


AUDIO_EXPORT_OPTIONS = {
    "opus": {"format": "opus", "bitrate": "32k"},
    "pcm": {"format": "s16le"},
    "wav": {"format": "wav"},  # 码率固定为 (采样率*位深度*声道数/1000)
    "mp3": {"format": "mp3", "bitrate": "128k"},
    "flac": {"format": "flac"},
    "aac": {"format": "adts", "codec": "aac", "bitrate": "128k"},
    "m4a": {"format": "ipod", "codec": "aac", "bitrate": "128k"},
}


def ndarray_to_any_format(
    audio_ndarray: np.ndarray,
    sample_rate=16000,
    resample_rate=16000,
    format: Literal["opus", "pcm", "wav", "mp3", "flac", "aac", "m4a"] = "opus",
    format_options: dict | None = None,
    return_base64=False,
):
    if format_options is None:
        format_options = {}
    audio_segment = ndarray_to_pydub(to_mono(audio_ndarray), sample_rate)
    if resample_rate != sample_rate:
        audio_segment = audio_segment.set_frame_rate(resample_rate)
    buffer = BytesIO()
    export_options = AUDIO_EXPORT_OPTIONS[format]
    export_options.update(**format_options)
    audio_segment.export(buffer, **export_options)
    audio_bytes = buffer.getvalue()
    if return_base64:
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return audio_base64
    return audio_bytes


def any_format_to_ndarray(
    audio_bytes_or_base64: bytes | str,
    format="opus",
    sample_rate=16000,
    resample_rate=16000,
):
    if isinstance(audio_bytes_or_base64, str):
        audio_bytes = base64.b64decode(audio_bytes_or_base64.encode("utf-8"))
    else:
        audio_bytes = audio_bytes_or_base64
    if format == "pcm":
        audio_bytes = add_wav_header(audio_bytes, sample_rate)
    with NamedTemporaryFile(suffix=f".{format}") as f:
        f.write(audio_bytes)
        f.flush()
        audio_ndarray, _ = librosa.load(f.name, sr=resample_rate, mono=True)
    return audio_ndarray


def format_ndarray_to_base64(audio_ndarray: np.ndarray, sample_rate: int, resample_rate: int, format: str):
    if format == "wav":
        return ndarray_to_base64_with_wav_header(audio_ndarray, sample_rate, resample_rate)
    return ndarray_to_any_format(audio_ndarray, sample_rate, resample_rate, format, return_base64=True)


def remove_silence(
    audio_ndarray: np.ndarray,
    sample_rate: int,
    left_retention_seconds=0,
    right_retention_seconds=0,
    top_db=60,
    frame_length=440,
    hop_length=220,
):
    _, (s, e) = librosa.effects.trim(audio_ndarray, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    s = int(s - left_retention_seconds * sample_rate)
    s = max(s, 0)
    e = int(e + right_retention_seconds * sample_rate)
    audio_ndarray = audio_ndarray[s:e]
    return audio_ndarray


PUNCTS = re.escape(string.punctuation + zhon.hanzi.punctuation)


def remove_puncts(text, mode: Literal["left", "right", "all"] = "all"):
    pattern = f"[{PUNCTS}]+"
    match mode:
        case "left":
            pattern = f"^{pattern}"
        case "right":
            pattern = f"{pattern}$"

    return re.sub(pattern, "", text)


class AudioEncoder:
    """
    ### 参数说明：
    mode：支持以下三种输出模式：
        - rawstream：纯裸流，每一帧都不带封装头，仅使用 `output_codec` 指定的编解码器进行编码
        - container：纯封装，每一帧都使用 `output_format` 指定的格式进行封装
        - mixed：混合模式，仅首帧带头，后续每一帧都是裸流
    output_format：输出音频封装格式
    output_codec：输出音频编解码格式
    output_sample_rate：输出音频采样率
    output_bit_rate：输出音频码率
    output_frame_ms：输出音频帧长
    """

    def __init__(
        self,
        mode: Literal["rawstream", "container", "mixed"] = "container",
        input_sample_rate=16000,
        output_format="wav",
        output_codec="pcm_s16le",
        output_sample_rate=16000,
        output_bit_rate=24000,
        output_frame_ms=60,
    ):
        self.mode = mode
        self.input_sample_rate = input_sample_rate
        self.output_format = output_format
        self.output_sample_rate = output_sample_rate
        self.output_bit_rate = output_bit_rate
        self.output_codec = output_codec
        self.output_frame_ms = output_frame_ms

        if self.mode == "mixed":
            self.buffer = io.BytesIO()
            self.container = av.open(
                self.buffer,
                mode="w",
                format=self.output_format,
                container_options={"live": "1", "cluster_time_limit": str(output_frame_ms)},  # ffmpeg -h muxer=webm
            )
            self.stream = self.container.add_stream(codec_name=self.output_codec)
            self.set_codec_context(self.stream.codec_context)
            self.stream.codec_context.options.update({"application": "lowdelay"})
            self.encode = self.encode_mixed
        elif self.mode == "rawstream":
            self.codec_ctx = av.CodecContext.create(output_codec, mode="w")
            self.set_codec_context(self.codec_ctx)
            self.codec_ctx.format = self.codec_ctx.codec.audio_formats[0]
            self.codec_ctx.open()
            self.resampler = av.AudioResampler(
                format=self.codec_ctx.format,
                layout=self.codec_ctx.layout,
                rate=self.codec_ctx.sample_rate,
            )
            self.encode = self.encode_rawstream
        else:
            self.encode = self.encode_container

    def close(self):
        if self.mode == "mixed":
            self.container.close()

    def set_codec_context(self, codec_ctx):
        codec_ctx.sample_rate = self.output_sample_rate
        codec_ctx.layout = "mono"
        codec_ctx.bit_rate = self.output_bit_rate
        codec_ctx.options.update({"vbr": "off", "frame_duration": str(self.output_frame_ms)})  # ffmpeg -h encoder=libopus

    def bytes_to_base64(self, audio_bytes: bytes):
        return base64.b64encode(audio_bytes).decode("utf-8")

    def ndarray_to_frame(self, audio_ndarray: np.ndarray):
        frame = av.AudioFrame.from_ndarray(audio_ndarray.reshape(1, -1), format="fltp", layout="mono")
        frame.sample_rate = self.input_sample_rate
        return frame

    def encode_mixed(self, audio_ndarray):
        frame = self.ndarray_to_frame(audio_ndarray)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        audio_bytes = self.buffer.getvalue()
        audio_base64 = self.bytes_to_base64(audio_bytes)
        self.buffer.seek(0)
        self.buffer.truncate()
        return audio_base64

    def encode_container(self, audio_ndarray):
        frame = self.ndarray_to_frame(audio_ndarray)
        buffer = io.BytesIO()
        with av.open(buffer, mode="w", format=self.output_format) as container:
            stream = container.add_stream(codec_name=self.output_codec)
            self.set_codec_context(stream.codec_context)
            packets = stream.encode(frame) + stream.encode(None)
            for packet in packets:
                container.mux(packet)
        return self.bytes_to_base64(buffer.getvalue())

    def encode_rawstream(self, audio_ndarray):
        frame = self.ndarray_to_frame(audio_ndarray)
        output_bytes = b""
        for resampled_frame in self.resampler.resample(frame):
            for packet in self.codec_ctx.encode(resampled_frame):
                output_bytes += bytes(packet)
        return self.bytes_to_base64(output_bytes)


def get_av_audio_encoder(input_sample_rate, output_sample_rate, output_format):
    FORMAT_TO_ENCODER_KWARGS = {
        "opus": {"mode": "container", "output_format": "ogg", "output_codec": "libopus"},
        "pcm": {"mode": "rawstream", "output_format": "wav", "output_codec": "pcm_s16le"},
        "wav": {"mode": "container", "output_format": "wav", "output_codec": "pcm_s16le"},
        "mp3": {"mode": "container", "output_format": "mp3", "output_codec": "libmp3lame"},
        "flac": {"mode": "container", "output_format": "flac", "output_codec": "flac"},
        "aac": {"mode": "container", "output_format": "adts", "output_codec": "aac"},
        "m4a": {"mode": "container", "output_format": "ipod", "output_codec": "aac"},
    }
    factor_kwargs = {
        "input_sample_rate": input_sample_rate,
        "output_sample_rate": output_sample_rate,
        "output_bit_rate": 32000,
        "output_frame_ms": 60,
    }
    encoder_kwargs = FORMAT_TO_ENCODER_KWARGS.get(output_format, FORMAT_TO_ENCODER_KWARGS["pcm"])
    encoder_kwargs.update(**factor_kwargs)
    encoder = AudioEncoder(**encoder_kwargs)
    return encoder
