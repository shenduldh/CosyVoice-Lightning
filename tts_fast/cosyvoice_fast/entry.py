import os
from typing import Optional, Union, Dict, List, Any, AsyncGenerator
from collections import OrderedDict
from functools import lru_cache
import torch
import uuid
import uvloop
import queue
import threading

from .pipeline import CosyVoicePipeline
from .frontend import CosyVoiceFrontEnd
from .common import VERSION, TTS_MODEL_DIR, CosyVoiceInputType, Prompt, Params


def async_to_sync_gen(async_generator: AsyncGenerator):
    q = queue.Queue()

    async def do_async_generation():
        try:
            async for i in async_generator:
                q.put(i)
        finally:
            q.put(None)

    def run_in_new_loop():
        loop = uvloop.new_event_loop()
        loop.run_until_complete(do_async_generation())

    threading.Thread(target=run_in_new_loop).start()

    while True:
        output = q.get_nowait() if not q.empty() else q.get()
        if output is None:
            break
        yield output


def clone_to_cpu(d: dict):
    _d = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.clone().cpu()
        _d[k] = v
    return _d


def tensor_to_list(t: torch.Tensor):
    return t.cpu().numpy().flatten().tolist()


class CosyVoiceEntry:
    def __init__(self):
        self.synthesizer = CosyVoicePipeline(TTS_MODEL_DIR)
        self.sample_rate = self.synthesizer.sample_rate
        self.frontend = CosyVoiceFrontEnd(TTS_MODEL_DIR)
        self.speaker_cache = OrderedDict()
        self.set_decoration()

    def save_cache(
        self,
        cache_dir: str,
        filename: Union[str, None] = None,
        speaker_ids: List[str] = [],
    ):
        if len(speaker_ids) > 0:
            saved = {k: v for k, v in self.speaker_cache.items() if k in speaker_ids}
        else:
            saved = {k: v for k, v in self.speaker_cache.items()}

        if len(saved) == 0:
            return None

        os.makedirs(cache_dir, exist_ok=True)
        if filename is None:
            filename = f"speaker_cache_{uuid.uuid4().hex}.pt"
        else:
            filename = f"{filename}.pt"
        cache_path = os.path.join(cache_dir, filename)
        if os.path.exists(cache_path):
            os.remove(cache_path)
        torch.save(saved, cache_path)
        return cache_path

    def load_cache(self, cache_path: str, speaker_ids: List[str] = []):
        if not os.path.exists(cache_path):
            return []
        loaded_cache = torch.load(cache_path, map_location=torch.device("cpu"), weights_only=False)
        if len(speaker_ids) > 0:
            filtered = {k: v for k, v in loaded_cache.items() if k in speaker_ids}
        else:
            filtered = loaded_cache
        self.speaker_cache.update(filtered)
        return list(filtered.keys())

    def get_speakers(self):
        return list(self.speaker_cache.keys())

    def remove_speakers(self, speaker_ids=[]):
        if len(speaker_ids) == 0:
            speaker_ids = list(self.speaker_cache.keys())

        removed = []
        for spk_id in speaker_ids:
            if spk_id in self.speaker_cache:
                self.speaker_cache.pop(spk_id)
                removed.append(spk_id)
        return removed

    def get_speech_features(self, audio_path, loudness):
        audio_tensor, sr = self.frontend.load_audio(audio_path, loudness)

        spk_embedding = self.frontend.extract_spk_embedding(audio_tensor, sr)
        mels, mels_lens = self.frontend.extract_speech_mels(audio_tensor, sr)
        tokens, tokens_lens = self.frontend.extract_speech_tokens(audio_tensor, sr)

        ## force len(mels) / len(tokens) == 2
        # calc real lengths
        tokens_lens, _ = torch.min(torch.stack(((mels_lens / 2).long(), tokens_lens), dim=1), dim=1)
        mels_lens = tokens_lens * 2
        # get max lengths
        max_tokens_len = torch.max(tokens_lens)
        max_mels_len = torch.max(mels_lens)
        # get nonempty masks
        nonempty_tokens_mask = torch.zeros(tokens.shape[0], max_tokens_len, *tokens.shape[2:])
        for i, l in enumerate(tokens_lens):
            nonempty_tokens_mask[i, :l] = 1
        nonempty_mels_mask = torch.zeros(mels.shape[0], max_mels_len, *mels.shape[2:])
        for i, l in enumerate(mels_lens):
            nonempty_mels_mask[i, :l] = 1
        # get aligned mels and tokens
        tokens = (tokens[:, :max_tokens_len] * nonempty_tokens_mask).long()
        mels = mels[:, :max_mels_len] * nonempty_mels_mask

        return {
            "speaker_embedding": spk_embedding,
            "speech_tokens": tokens,
            "speech_mels": mels,
        }

    @lru_cache(maxsize=128)
    def get_text_features(self, text) -> Dict[str, List[int]]:
        text_tokens = self.frontend.extract_text_tokens(text)
        return {"text_tokens": text_tokens}

    def get_speaker_features(
        self,
        prompt_audio: Optional[str] = None,
        prompt_text: Optional[str] = None,
        speaker_id: Optional[str] = None,
        loudness: float = 30.0,
    ):
        # get speaker features from cache
        if speaker_id and speaker_id in self.speaker_cache:
            return self.speaker_cache[speaker_id]

        # calc speaker features
        speaker_features = {}
        # get speech features
        speech_features = self.get_speech_features(prompt_audio, loudness)
        speaker_features.update(speech_features)
        # get text features
        prompt_text = self.clean_prompt_decoration(prompt_text)
        text_features = self.get_text_features(prompt_text)
        speaker_features.update(text_features)
        # cache speaker features
        if speaker_id:
            self.speaker_cache[speaker_id] = clone_to_cpu(speaker_features)

        return speaker_features

    def set_decoration(self):
        if VERSION == "cosyvoice3":
            prompt_prefix = "You are a helpful assistant.<|endofprompt|>"
            prompt_suffix = ""
            instruct_prefix = "You are a helpful assistant. "
            instruct_suffix = "<|endofprompt|>"
        else:
            prompt_prefix = prompt_suffix = instruct_prefix = ""
            instruct_suffix = "<|endofprompt|>"

        self.prompt_prefix_tokens = self.get_text_features(prompt_prefix)["text_tokens"]
        self.prompt_suffix_tokens = self.get_text_features(prompt_suffix)["text_tokens"]
        self.instruct_prefix_tokens = self.get_text_features(instruct_prefix)["text_tokens"]
        self.instruct_suffix_tokens = self.get_text_features(instruct_suffix)["text_tokens"]

        self.clean_prompt_decoration = lambda t: t.strip().removeprefix(prompt_prefix).removeprefix(prompt_suffix).strip()
        self.clean_instruct_decoration = lambda t: t.strip().removeprefix(instruct_prefix).removesuffix(instruct_suffix).strip()

    def prepare_prompt(
        self,
        prompt_audio: Optional[str] = None,
        prompt_text: Optional[str] = None,
        instruct_text: Optional[str] = None,
        speaker_id: Optional[str] = None,
        loudness: float = 30.0,
    ) -> Prompt:
        # get speaker features
        speaker_features = self.get_speaker_features(prompt_audio, prompt_text, speaker_id, loudness)

        # create prompt
        prompt = Prompt(
            speaker_id=speaker_id or uuid.uuid4().hex,
            speaker_embedding=speaker_features["speaker_embedding"],
            speech_mels=speaker_features["speech_mels"],
            flow_speech_tokens=speaker_features["speech_tokens"],
            llm_speech_tokens=tensor_to_list(speaker_features["speech_tokens"]),
            text_tokens=speaker_features["text_tokens"],
            prefix_tokens=self.prompt_prefix_tokens,
            suffix_tokens=self.prompt_suffix_tokens,
        )

        if instruct_text is not None:
            # get instruct text features
            instruct_text = self.clean_instruct_decoration(instruct_text)
            prompt.text_tokens = self.get_text_features(instruct_text)["text_tokens"]
            prompt.prefix_tokens = self.instruct_prefix_tokens
            prompt.suffix_tokens = self.instruct_suffix_tokens
            prompt.llm_speech_tokens = []

        return prompt

    def async_request(
        self,
        tts_text: Optional[Any],
        prompt_audio: Optional[str],
        prompt_text: Optional[str],
        instruct_text: Optional[str],
        speaker_id: Optional[str] = None,
        loudness=30.0,
        split_text=False,
        stream=True,
        generation_params: dict = {},
        input_type=CosyVoiceInputType.SINGLE,
    ):
        prompt = self.prepare_prompt(prompt_audio, prompt_text, instruct_text, speaker_id, loudness)
        if tts_text is None:
            return

        params = Params(stream, **generation_params)
        input_generator = self.wrap_to_generator(tts_text, split_text, input_type)
        output_generator = self.synthesizer.generate(input_generator=input_generator, prompt=prompt, params=params)
        return output_generator

    def request(self, *args, **kwargs):
        return async_to_sync_gen(self.async_request(*args, **kwargs))

    async def preprocess_text(self, text, split_text):
        res = self.frontend.normalize_text(text, split_text, True)
        if res is None:
            res = []
        if not isinstance(res, list):
            res = [res]
        for i in res:
            text_tokens = self.frontend.extract_text_tokens(i)
            yield text_tokens

    async def wrap_to_generator(self, obj, split_text, input_type=CosyVoiceInputType.SINGLE):
        if input_type == CosyVoiceInputType.SINGLE:
            # normalized -> token ids, Generator, AsyncGenerator
            async for normalized in self.preprocess_text(obj, split_text):
                yield normalized
        elif input_type == CosyVoiceInputType.GENERATOR:
            async for i in obj:
                async for j in self.wrap_to_generator(i, split_text):
                    yield j
        elif input_type == CosyVoiceInputType.QUEUE:
            while True:
                i = await obj.get()
                if i is None:
                    break
                async for j in self.wrap_to_generator(i, split_text):
                    yield j
