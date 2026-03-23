from typing import Iterable, Optional, Tuple
import torch
from torch import nn
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix
from sglang.srt.models.qwen2 import Qwen2Model


class CosyVoice3LLM(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        self.hidden_size = 896
        self.num_text_tokens = 151936
        self.num_speech_tokens = 6561
        # stop_token_ids: num_speech_tokens ~ num_speech_tokens + 199
        self.num_stop_tokens = 200
        # sos_token_id: num_speech_tokens
        # eos_token_id: num_speech_tokens + 1
        # task_token_id: num_speech_tokens + 2
        # fill_token_id: num_speech_tokens + 3
        # included in the stop tokens
        self.num_special_tokens = 4
        self.num_generation_tokens = self.num_speech_tokens + self.num_stop_tokens

        self.config.vocab_size = self.num_text_tokens
        self.model = Qwen2Model(
            self.config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.config.vocab_size = self.num_generation_tokens
        self.logits_processor = LogitsProcessor(self.config)

        # used to embed speech tokens, special tokens and stop tokens
        self.speech_embedding = torch.nn.Embedding(
            num_embeddings=self.num_generation_tokens,
            embedding_dim=self.hidden_size,
        )
        # used to generate logits regarding speech embeddings
        self.llm_decoder = ParallelLMHead(
            num_embeddings=self.num_generation_tokens,
            embedding_dim=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("llm_decoder", prefix),
        )

        self.mixed_embedding = VocabParallelEmbedding(
            num_embeddings=self.num_generation_tokens + self.num_text_tokens,
            embedding_dim=self.hidden_size,
            quant_config=quant_config,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.mixed_embedding(input_ids)

        hidden_states = self.model(
            None,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.llm_decoder, forward_batch
            )
        else:
            return hidden_states

    def convert_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        for name, loaded_weight in weights:
            if name.startswith(
                ("llm.model.model.", "speech_embedding.", "llm_decoder.")
            ):
                if name.startswith("llm.model.model."):
                    name = name.replace("llm.model.model.", "model.")
                yield name, loaded_weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        weights = self.convert_weights(weights)
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        device = self.speech_embedding.weight.device
        speech_embedding = self.speech_embedding.weight
        text_embedding = self.model.embed_tokens(
            torch.arange(self.num_text_tokens).to(device)
        )
        concatenated = torch.cat([speech_embedding, text_embedding])
        self.mixed_embedding.weight_loader(self.mixed_embedding.weight, concatenated)
        del self.speech_embedding
        del self.model.embed_tokens


EntryClass = CosyVoice3LLM
