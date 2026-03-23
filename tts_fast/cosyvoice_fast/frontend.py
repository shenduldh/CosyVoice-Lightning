from functools import partial
from typing import Generator, AsyncGenerator, Tuple
import json
import onnxruntime
import torch
import numpy as np
import whisper
import torchaudio.compliance.kaldi as kaldi
import os
import re
import inflect
import librosa
import soundfile
import torchaudio
from loguru import logger
# import s3tokenizer

from cosyvoice.utils.frontend_utils import (
    contains_chinese,
    replace_blank,
    replace_corner_mark,
    remove_bracket,
    spell_out_number,
    split_paragraph,
    is_only_punctuation,
)
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
from matcha.utils.audio import mel_spectrogram
from NovaSR import FastSR
from .common import VERSION, TTSFRD_RESOURCE_PATH, FRONTEND_MODE, NOVASR_MODEL_PATH


def get_mel_encoder():
    if VERSION == "cosyvoice3":
        mel_encoder = lambda y: mel_spectrogram(
            y,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=None,
            center=False,
        )
    else:
        mel_encoder = lambda y: mel_spectrogram(
            y,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=8000,
            center=False,
        )
    return mel_encoder, 24000


def get_tokenizer():
    qwen_config_path = os.path.join(os.path.dirname(__file__), "llm", "qwen2_config")

    if VERSION == "cosyvoice3":
        return get_qwen_tokenizer(
            qwen_config_path, skip_special_tokens=True, version="cosyvoice3"
        )
    return get_qwen_tokenizer(
        qwen_config_path, skip_special_tokens=True, version="cosyvoice2"
    )


def get_speech_tokenizer(model_dir, device):
    speech_tokenizer_path = os.path.join(
        model_dir,
        "speech_tokenizer_v3.onnx"
        if VERSION == "cosyvoice3"
        else "speech_tokenizer_v2.onnx",
    )

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    speech_tokenizer = onnxruntime.InferenceSession(
        speech_tokenizer_path,
        sess_options=option,
        providers=[
            (
                "CUDAExecutionProvider"
                if torch.cuda.is_available()
                else "CPUExecutionProvider"
            )
        ],
    )

    # speech_tokenizer = s3tokenizer.load_model(speech_tokenizer_path).to(device).eval()

    return speech_tokenizer, 16000


def get_speaker_encoder(model_dir):
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1

    speaker_encoder_path = os.path.join(model_dir, "campplus.onnx")
    speaker_encoder = onnxruntime.InferenceSession(
        speaker_encoder_path, sess_options=option, providers=["CPUExecutionProvider"]
    )

    return speaker_encoder, 16000


class CosyVoiceFrontEnd:
    def __init__(self, model_dir, allowed_special="all"):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.tokenizer = get_tokenizer()
        self.mel_encoder, self.mel_feature_sr = get_mel_encoder()
        self.speaker_encoder, self.speaker_embedding_sr = get_speaker_encoder(model_dir)
        self.speech_tokenizer, self.speech_token_sr = get_speech_tokenizer(
            model_dir, self.device
        )

        self.allowed_special = allowed_special
        self.use_ttsfrd = FRONTEND_MODE == "ttsfrd"
        if self.use_ttsfrd:
            import ttsfrd

            self.ttsfrd = ttsfrd.TtsFrontendEngine()
            assert self.ttsfrd.initialize(TTSFRD_RESOURCE_PATH) is True, (
                "Failed to initialize ttsfrd resource."
            )
            self.ttsfrd.set_lang_type("pinyinvg")
        else:
            import wetext

            self.wetext = wetext.Normalizer()
            self.inflect_parser = inflect.engine()
            self.tokenizer_encode_fn = partial(
                self.tokenizer.encode, allowed_special=allowed_special
            )

        if NOVASR_MODEL_PATH is not None:
            self.upsampler = FastSR(ckpt_path=NOVASR_MODEL_PATH, half=False)
        else:
            self.upsampler = None

        logger.info(f"{self.__class__.__name__} is ready.")

    def __ett_generator(self, text_generator: Generator):
        for text in text_generator:
            text_token = self.extract_text_tokens(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i : i + 1]

    async def __ett_async_generator(self, text_generator: AsyncGenerator):
        async for text in text_generator:
            text_token = self.extract_text_tokens(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i : i + 1]

    def extract_text_tokens(self, text):
        if isinstance(text, Generator):
            return self.__ett_generator(text)
        elif isinstance(text, AsyncGenerator):
            return self.__ett_async_generator(text)
        else:
            token_ids = self.tokenizer.encode(
                text, allowed_special=self.allowed_special
            )
            return token_ids

    def extract_speech_tokens(self, audio_tensor, sr: int):
        assert audio_tensor.shape[1] / sr <= 30, (
            "Do not support extract speech tokens for audio longer than 30s."
        )
        if sr != self.speech_token_sr:
            audio_tensor = self.resample(audio_tensor, sr, self.speech_token_sr)

        # mel = s3tokenizer.log_mel_spectrogram(audio_tensor[0], n_mels=128)
        # mels, mel_lens = s3tokenizer.padding([mel])
        # speech_tokens, speech_tokens_lens = self.speech_tokenizer.quantize(
        #     mels.to(self.device), mel_lens.to(self.device)
        # )
        # return speech_tokens, speech_tokens_lens

        feat = whisper.log_mel_spectrogram(audio_tensor, n_mels=128)
        input_name0 = self.speech_tokenizer.get_inputs()[0].name
        input_name1 = self.speech_tokenizer.get_inputs()[1].name
        speech_tokens = self.speech_tokenizer.run(
            None,
            {
                input_name0: feat.cpu().numpy(),
                input_name1: np.array([feat.shape[2]], dtype=np.int32),
            },
        )
        speech_tokens = speech_tokens[0].flatten().tolist()
        speech_tokens = torch.tensor([speech_tokens], dtype=torch.int32).cpu()
        speech_tokens_lens = torch.tensor(
            [speech_tokens.shape[1]], dtype=torch.int32
        ).cpu()
        return speech_tokens, speech_tokens_lens

    def extract_spk_embedding(self, audio_tensor, sr: int):
        if sr != self.speaker_embedding_sr:
            audio_tensor = self.resample(audio_tensor, sr, self.speaker_embedding_sr)

        feat = kaldi.fbank(
            audio_tensor, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        input_name = self.speaker_encoder.get_inputs()[0].name
        embedding = self.speaker_encoder.run(
            None, {input_name: feat.unsqueeze(dim=0).cpu().numpy()}
        )
        embedding = torch.from_numpy(embedding[0]).cpu()
        return embedding

    def extract_speech_mels(self, audio_tensor, sr: int):
        if sr != self.mel_feature_sr:
            audio_tensor = self.resample(audio_tensor, sr, self.mel_feature_sr)

        mels = self.mel_encoder(audio_tensor)
        mels = mels.squeeze(dim=0).transpose(0, 1).unsqueeze(dim=0).cpu()
        mels_lens = torch.tensor([mels.shape[1]], dtype=torch.int32).cpu()
        return mels, mels_lens

    def normalize_text(
        self, text, split=False, enabled=True
    ) -> AsyncGenerator | str | list[str] | None:
        if isinstance(text, AsyncGenerator):
            return text

        if not enabled or text == "" or ("<|" in text and "|>" in text):
            return text if not split else [text]

        if self.use_ttsfrd:
            res = json.loads(self.ttsfrd.do_voicegen_frd(text))["sentences"]
            if res is None:
                return None
            text = [i["text"] for i in res]
            if not split:
                text = "".join(text)
        else:
            lang = "zh" if contains_chinese(text) else "en"

            text = self.wetext.normalize(text, lang=lang)
            if lang == "zh":
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r"[，,、]+$", "。", text)
            else:
                text = spell_out_number(text, self.inflect_parser)

            if split:
                text = split_paragraph(
                    text,
                    self.tokenizer_encode_fn,
                    lang,
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )

        if split:
            text = [i for i in text if not is_only_punctuation(i)]
        return text

    def load_audio(
        self,
        path: str,
        target_loudness=30.0,  # normalize loudness in dB (LKFS)
        max_amplification=0.8,
        tail_silence_seconds=0.2,
        upsample_sr_threshold=24000,
    ) -> Tuple[torch.Tensor, int]:
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)  # to mono

        # remove silence
        audio_tensor, _ = librosa.effects.trim(
            audio_tensor, top_db=60, frame_length=440, hop_length=220
        )
        # audio_tensor = torchaudio.functional.vad(audio_tensor, sr)

        # concat tail silence
        tail_silence = torch.zeros(1, int(sr * tail_silence_seconds)).float()
        audio_tensor = torch.cat([audio_tensor, tail_silence], dim=1)

        # normalize loudness
        orig_loudness = torchaudio.functional.loudness(audio_tensor, sr).item()
        delta_loudness = target_loudness - orig_loudness
        gain = 10.0 ** (delta_loudness / 20.0)
        audio_tensor *= gain
        if audio_tensor.abs().max() > max_amplification:
            audio_tensor = audio_tensor / audio_tensor.abs().max() * max_amplification

        # upsample audio to 48000 sample rate
        if self.upsampler and sr < upsample_sr_threshold:
            try:
                resampled = self.resample(audio_tensor, sr, 16000, "kaiser_window")
                resampled = resampled.unsqueeze(1).to(self.upsampler.device)
                audio_tensor = self.upsampler.infer(resampled).cpu()
                sr = 48000
            except:
                pass

        return audio_tensor, sr

    def resample(
        self, audio_tensor, orig_sr, target_sr, resampling_method="sinc_interp_hann"
    ):
        audio_tensor = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=target_sr, resampling_method=resampling_method
        )(audio_tensor)
        return audio_tensor

    def save_audio(self, saved_path, audio_ndarray, sample_rate):
        soundfile.write(saved_path, audio_ndarray, sample_rate)
