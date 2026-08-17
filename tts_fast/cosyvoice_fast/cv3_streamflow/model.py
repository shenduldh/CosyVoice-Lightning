from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import math
import random
import numpy as np
from x_transformers.x_transformers import RotaryEmbedding, rotate_half


def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, orig_dtype = freqs.shape[-1], t.dtype

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t, t_unrotated), dim=-1)

    return out.type(orig_dtype)


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,  # noised input x
        mask: Optional[torch.Tensor] = None,
        rope=None,  # rotary position embedding
        key_cache=None,
        value_cache=None,
        return_cache=False,
    ):
        batch_size = x.shape[0]

        # `sample` projections.
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # apply rotary position embedding
        if rope is not None:
            query = apply_rotary_pos_emb(query, rope, scale=1)
            key = apply_rotary_pos_emb(key, rope, scale=1)

        # attention
        query = query.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

        if return_cache:
            new_key_cache = key
            new_value_cache = value
        else:
            new_key_cache = new_value_cache = None

        if key_cache is not None:
            key = torch.concat((key_cache, key), dim=-2)
            value = torch.concat((value_cache, value), dim=-2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        attn_mask = None if mask is None else mask
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)

        # linear proj
        x = self.to_out[0](x)
        # dropout
        x = self.to_out[1](x)

        # if mask is not None:
        #     mask = torch.diagonal(mask[:, 0], dim1=-2, dim2=-1).unsqueeze(-1)
        #     x = x.masked_fill(~mask, 0.0)

        return x, new_key_cache, new_value_cache


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None, key_cache=None, value_cache=None, return_cache=False):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output, new_key_cache, new_value_cache = self.attn(norm, mask, rope, key_cache, value_cache, return_cache)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(ff_norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x, new_key_cache, new_value_cache


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time


class CausalConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, conv_cache=None, return_cache=False):
        x = x.permute(0, 2, 1)

        new_conv_cache = [] if return_cache else None

        if conv_cache is not None:
            x = torch.cat((conv_cache[0], x), dim=-1)
        else:
            x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))

        if return_cache:
            new_conv_cache.append(x[:, :, -self.kernel_size + 1 :])

        x = self.conv1(x)

        if conv_cache is not None:
            x = torch.cat((conv_cache[1], x), dim=-1)
        else:
            x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))

        if return_cache:
            new_conv_cache.append(x[:, :, -self.kernel_size + 1 :])

        x = self.conv2(x)
        out = x.permute(0, 2, 1)

        if return_cache:
            new_conv_cache = torch.stack(new_conv_cache)

        return out, new_conv_cache


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, spk_dim=None):
        super().__init__()
        spk_dim = 0 if spk_dim is None else spk_dim
        self.spk_dim = spk_dim
        self.proj = nn.Linear(mel_dim * 2 + text_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text_embed: torch.Tensor,
        spks: torch.Tensor,
        conv_cache=None,
        return_cache=False,
    ):
        x = self.proj(torch.concat([x, cond, text_embed, spks], dim=-1))
        conv_out, new_conv_cache = self.conv_pos_embed(x, conv_cache, return_cache)
        x = conv_out + x
        return x, new_conv_cache


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        mu_dim=None,
        long_skip_connection=False,
        spk_dim=None,
        out_channels=None,
        static_chunk_size=50,
        num_decoding_left_chunks=2,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if mu_dim is None:
            mu_dim = mel_dim
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)
        self.out_channels = out_channels
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    # def forward(self, x, mu, t, spks, cond, mask, offset, conv_cache=None, attn_k_cache=None, attn_v_cache=None, return_cache=False):
    def forward(self, x, mu, t, spks, cond, offset, conv_cache=None, attn_k_cache=None, attn_v_cache=None, return_cache=False):
        btz, seq_len, device = x.shape[0], x.shape[1], x.device

        rope_indexes = torch.arange(seq_len, device=device) + offset
        rope, _ = self.rotary_embed.forward(rope_indexes)

        if attn_k_cache is not None:
            cache_len = attn_k_cache.shape[-2]
            kv_len = cache_len + seq_len
            r_idx = torch.arange(seq_len, device=device) + offset
            block_value_r = (r_idx // self.static_chunk_size + 1) * self.static_chunk_size
            c_idx = torch.arange(kv_len, device=device) + (offset - cache_len)
            attn_mask = c_idx.unsqueeze(0) < block_value_r.unsqueeze(1)  # (seq_len, kv_len)
        else:
            pos_idx = torch.arange(seq_len, device=device)
            block_value = (pos_idx // self.static_chunk_size + 1) * self.static_chunk_size
            attn_mask = pos_idx.unsqueeze(0) < block_value.unsqueeze(1)  # (seq_len, seq_len)

        attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
        attn_mask = attn_mask.expand(btz, -1, -1, -1)

        # attn_mask = attn_mask & mask.unsqueeze(-1)
        # attn_mask[:, :, :, -seq_len:] = attn_mask[:, :, :, -seq_len:] & mask.unsqueeze(2)
        # if (attn_mask.sum(dim=-1) == 0).sum().item() != 0:
        #     attn_mask[attn_mask.sum(dim=-1) == 0] = True

        x, new_conv_cache = self.input_embed(x, cond, mu, spks, conv_cache, return_cache)
        t = self.time_embed(t)

        if self.long_skip_connection is not None:
            residual = x

        if return_cache:
            new_attn_k_cache = []
            new_attn_v_cache = []
        else:
            new_attn_k_cache = new_attn_v_cache = None

        for i, block in enumerate(self.transformer_blocks):
            if attn_k_cache is not None:
                this_k_cache = attn_k_cache[i]
                this_v_cache = attn_v_cache[i]
            else:
                this_k_cache = this_v_cache = None

            x, this_new_k_cache, this_new_v_cache = block(
                x,
                t,
                mask=attn_mask,
                rope=rope,
                key_cache=this_k_cache,
                value_cache=this_v_cache,
                return_cache=return_cache,
            )

            if return_cache:
                new_attn_k_cache.append(this_new_k_cache)
                new_attn_v_cache.append(this_new_v_cache)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        if return_cache:
            new_attn_k_cache = torch.stack(new_attn_k_cache)
            new_attn_v_cache = torch.stack(new_attn_v_cache)

        return output, new_conv_cache, new_attn_k_cache, new_attn_v_cache


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CausalConditionalCFM(nn.Module):
    def __init__(self, in_channels, cfm_params, n_spks, spk_emb_dim, estimator: DiT):
        super().__init__()
        set_all_random_seed(0)
        self.n_feats = in_channels
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min if hasattr(cfm_params, "sigma_min") else 1e-4
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.estimator = estimator
        self.register_buffer("rand_noise", torch.randn([1, 50 * 300, 80]))

        self.use_trt_estimator = False
        self.trt_estimator = None
        self.trt_estimator_cache_in = None
        self.trt_estimator_cache_out = None
        self.trt_estimator_cico = None

    # def forward(
    #     self, mu, mask, spks, cond, n_timesteps, offset, temperature=1.0, conv_cache=None, attn_k_cache=None, attn_v_cache=None, return_cache=False
    # ):
    def forward(
        self, mu, spks, cond, n_timesteps, offset, temperature=1.0, conv_cache=None, attn_k_cache=None, attn_v_cache=None, return_cache=False
    ):
        """
        mu: [batch_size, seq_len, n_feats]
        mask: [batch_size, 1, seq_len]
        return: [batch_size, n_feats, seq_len]
        """
        z = self.rand_noise[:, : mu.size(1), :] * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(
            z,
            t_span=t_span,
            mu=mu,
            # mask=mask,
            spks=spks,
            cond=cond,
            offset=offset,
            conv_cache=conv_cache,
            attn_k_cache=attn_k_cache,
            attn_v_cache=attn_v_cache,
            return_cache=return_cache,
        )

    @torch.compiler.disable
    @torch.amp.autocast("cuda", enabled=False)
    def call_trt_estimator(self, x_in, mu_in, t_in, spks_in, cond_in, offset, conv_cache, attn_k_cache, attn_v_cache, return_cache):
        new_conv_cache = new_attn_k_cache = new_attn_v_cache = None
        if conv_cache is not None:
            inputs = (x_in, mu_in, t_in, spks_in, cond_in, offset, conv_cache, attn_k_cache, attn_v_cache)
            if return_cache:
                output, new_conv_cache, new_attn_k_cache, new_attn_v_cache = self.trt_estimator_cico(*inputs)
            else:
                output = self.trt_estimator_cache_in(*inputs)
        else:
            inputs = (x_in, mu_in, t_in, spks_in, cond_in)
            if return_cache:
                output, new_conv_cache, new_attn_k_cache, new_attn_v_cache = self.trt_estimator_cache_out(*inputs)
            else:
                output = self.trt_estimator(*inputs)
        return output, new_conv_cache, new_attn_k_cache, new_attn_v_cache

    def solve_euler(
        self,
        x,
        t_span,
        mu,
        # mask,
        spks,
        cond,
        offset,
        conv_cache=None,
        attn_k_cache=None,
        attn_v_cache=None,
        return_cache=False,
    ):
        btz, device = x.shape[0], x.device

        offset = torch.tensor([offset], dtype=torch.int64, device=x.device)
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        if return_cache:
            new_conv_cache = []
            new_attn_k_cache = []
            new_attn_v_cache = []
        else:
            new_conv_cache = new_attn_k_cache = new_attn_v_cache = None

        for i in range(len(t_span) - 1):
            t = t_span[i]
            dt = t_span[i + 1] - t

            x_in = torch.cat([x, x], dim=0)
            t_in = t_span[i : i + 1].expand(btz * 2)

            if conv_cache is not None:
                this_conv_cache = conv_cache[i].to(device)
                this_attn_k_cache = attn_k_cache[i].to(device)
                this_attn_v_cache = attn_v_cache[i].to(device)
            else:
                this_conv_cache = this_attn_k_cache = this_attn_v_cache = None

            # estimator_inputs = (x_in, mu_in, t_in, spks_in, cond_in, mask, offset, this_conv_cache, this_attn_k_cache, this_attn_v_cache, return_cache)
            estimator_inputs = (x_in, mu_in, t_in, spks_in, cond_in, offset, this_conv_cache, this_attn_k_cache, this_attn_v_cache, return_cache)
            dphi_dt, this_new_conv_cache, this_new_attn_k_cache, this_new_attn_v_cache = (
                self.call_trt_estimator(*estimator_inputs) if self.use_trt_estimator else self.estimator(*estimator_inputs)
            )

            cond_dphi, uncond_dphi = torch.split(dphi_dt, btz, dim=0)
            dphi_dt = (1.0 + self.inference_cfg_rate) * cond_dphi - self.inference_cfg_rate * uncond_dphi
            x = x + dt * dphi_dt

            if return_cache:
                new_conv_cache.append(this_new_conv_cache)
                new_attn_k_cache.append(this_new_attn_k_cache)
                new_attn_v_cache.append(this_new_attn_v_cache)

        if return_cache:
            new_conv_cache = torch.stack(new_conv_cache)
            new_attn_k_cache = torch.stack(new_attn_k_cache)
            new_attn_v_cache = torch.stack(new_attn_v_cache)

        return x.transpose(1, 2), new_conv_cache, new_attn_k_cache, new_attn_v_cache


class PreLookaheadLayer(nn.Module):
    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1_kernel_size = pre_lookahead_len + 1
        self.conv2_kernel_size = 3
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=self.conv1_kernel_size, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channels, in_channels, kernel_size=self.conv2_kernel_size, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, finalize=False, cache=None, return_cache=False):
        """
        inputs: (batch_size, seq_len, text_embed_dim)
        """
        outputs = inputs.transpose(1, 2).contiguous()

        # look ahead
        if finalize:
            outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode="constant", value=0.0)
        else:
            inputs = inputs[:, : -self.pre_lookahead_len]
        outputs = F.leaky_relu(self.conv1(outputs))

        # look back
        pad_len = self.conv2_kernel_size - 1
        if cache is not None:
            outputs = torch.concat([cache.to(outputs.device), outputs], dim=2)
        else:
            outputs = F.pad(outputs, (pad_len, 0), mode="constant", value=0.0)
        new_cache = outputs[:, :, -pad_len:] if return_cache else None
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        # residual connection
        outputs = outputs + inputs

        return outputs, new_cache


class CausalMaskedDiffWithDiT(nn.Module):
    def __init__(
        self,
        input_size=80,
        output_size=80,
        spk_embed_dim=192,
        output_type="mel",
        vocab_size=6561,
        input_frame_rate=25,
        only_mask_loss=True,
        token_mel_ratio=2,
        pre_lookahead_len=3,
        pre_lookahead_layer=PreLookaheadLayer(in_channels=80, channels=1024, pre_lookahead_len=3),
        decoder=CausalConditionalCFM(
            in_channels=240,
            n_spks=1,
            spk_emb_dim=80,
            cfm_params=DictConfig(
                content={
                    "sigma_min": 1e-06,
                    "solver": "euler",
                    "t_scheduler": "cosine",
                    "training_cfg_rate": 0.2,
                    "inference_cfg_rate": 0.7,
                    "reg_loss_type": "l1",
                }
            ),
            estimator=DiT(
                dim=1024,
                depth=22,
                heads=16,
                dim_head=64,
                ff_mult=2,
                mel_dim=80,
                mu_dim=80,
                spk_dim=80,
                out_channels=80,
                static_chunk_size=25 * 2,
                num_decoding_left_chunks=-1,
            ),
        ),
        decoder_conf=dict(),
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio

    @torch.inference_mode()
    def inference(
        self,
        token,
        prompt_feat,
        embedding,
        finalize,
        cache_offset=None,
        pll_cache=None,
        dit_conv_cache=None,
        dit_attn_k_cache=None,
        dit_attn_v_cache=None,
        return_cache=False,
    ):
        """
        token = prompt_token + generated_token
        token size = [1, seq_len]
        """
        ## get text condition
        token = self.input_embedding(torch.clamp(token, min=0))
        h, new_pll_cache = self.pre_lookahead_layer(token, finalize, pll_cache, return_cache)
        # [1, seq_len, 80]
        h = h.repeat_interleave(self.token_mel_ratio, dim=1).contiguous()

        if cache_offset is None:
            cache_offset = 0 if dit_attn_k_cache is None else dit_attn_k_cache.shape[-2]

        ## get mel condition
        this_mel_len = h.shape[1]
        # [1, seq_len, 80]
        conds = torch.zeros([1, this_mel_len, self.output_size], dtype=h.dtype, device=h.device)
        filling_prompt_len = min(prompt_feat.shape[1] - cache_offset, this_mel_len)
        if filling_prompt_len > 0:
            conds[:, :filling_prompt_len, :] = prompt_feat[:, cache_offset : cache_offset + filling_prompt_len, :]
        # mask = torch.ones(1, 1, this_mel_len, dtype=torch.bool, device=h.device)  # [1, 1, seq_len]

        ## get speaker condition
        # [1, seq_len, 80]
        embedding = self.spk_embed_affine_layer(F.normalize(embedding, dim=1)).unsqueeze(1).expand(-1, this_mel_len, -1)

        # generate mel
        feat, new_dit_conv_cache, new_dit_attn_k_cache, new_dit_attn_v_cache = self.decoder(
            mu=h,
            # mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            offset=cache_offset,
            conv_cache=dit_conv_cache,
            attn_k_cache=dit_attn_k_cache,
            attn_v_cache=dit_attn_v_cache,
            return_cache=return_cache,
        )
        # remove prompt part
        if filling_prompt_len > 0:
            feat = feat[:, :, filling_prompt_len:]

        new_cache_offset = cache_offset + this_mel_len

        return feat, new_cache_offset, new_pll_cache, new_dit_conv_cache, new_dit_attn_k_cache, new_dit_attn_v_cache
