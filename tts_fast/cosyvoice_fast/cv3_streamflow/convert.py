import torch
from torch import nn
import torch_tensorrt
import gc
import os
from typing import List, Literal
from functools import partial

from .model import CausalMaskedDiffWithDiT, DiT


class WrappedDiTCacheInCacheOut(nn.Module):
    def __init__(self, dit_model: DiT):
        super().__init__()
        self.dit_model = dit_model

    def forward(self, x, mu, t, spks, cond, offset, conv_cache, attn_k_cache, attn_v_cache):
        output, new_conv_cache, new_attn_k_cache, new_attn_v_cache = self.dit_model.forward(
            x, mu, t, spks, cond, offset, conv_cache, attn_k_cache, attn_v_cache, return_cache=True
        )
        return output, new_conv_cache, new_attn_k_cache, new_attn_v_cache


class WrappedDiTCacheOut(nn.Module):
    def __init__(self, dit_model: DiT):
        super().__init__()
        self.dit_model = dit_model
        self.register_buffer("offset", torch.tensor([0]))

    def forward(self, x, mu, t, spks, cond):
        output, new_conv_cache, new_attn_k_cache, new_attn_v_cache = self.dit_model.forward(
            x, mu, t, spks, cond, self.offset, None, None, None, return_cache=True
        )
        return output, new_conv_cache, new_attn_k_cache, new_attn_v_cache


class WrappedDiTCacheIn(nn.Module):
    def __init__(self, dit_model: DiT):
        super().__init__()
        self.dit_model = dit_model

    def forward(self, x, mu, t, spks, cond, offset, conv_cache, attn_k_cache, attn_v_cache):
        output, *_ = self.dit_model.forward(x, mu, t, spks, cond, offset, conv_cache, attn_k_cache, attn_v_cache, return_cache=False)
        return output


class WrappedDiT(nn.Module):
    def __init__(self, dit_model: DiT):
        super().__init__()
        self.dit_model = dit_model
        self.register_buffer("offset", torch.tensor([0]))

    def forward(self, x, mu, t, spks, cond):
        output, *_ = self.dit_model.forward(x, mu, t, spks, cond, self.offset, None, None, None, return_cache=False)
        return output


def export_and_compile(model, compiled_type, arg_inputs, dynamic_shapes, dtype, precisions, workspace_size, device):
    model.eval().to(device, dtype)

    if compiled_type in ["cache-out", "no-cache"]:
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    ep = torch.export.export(model, args=arg_inputs, dynamic_shapes=dynamic_shapes)
    if compiled_type in ["cache-out", "no-cache"]:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    print("#### Export is successful.")

    gc.collect()
    torch.cuda.empty_cache()

    # compile model
    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        arg_inputs=arg_inputs,
        workspace_size=workspace_size,
        enabled_precisions=precisions,
        use_strong_typing=True,
        enable_experimental_decompositions=True,
        assume_dynamic_shape_support=True,
        require_full_compilation=True,
    )

    outputs = trt_model(*arg_inputs)
    print(f"#### Test model: {outputs=}")

    gc.collect()
    torch.cuda.empty_cache()

    return trt_model


def build_mapping(dtype, device):
    seq_len = torch.export.Dim("seq_len", min=1, max=3000)
    attn_cache_len = torch.export.Dim("attn_cache_len", min=1, max=3000)
    dynamic_shapes_cache = {
        "x": {1: seq_len},
        "mu": {1: seq_len},
        "t": {},
        "spks": {1: seq_len},
        "cond": {1: seq_len},
        "offset": {},
        "conv_cache": {},
        "attn_k_cache": {3: attn_cache_len},
        "attn_v_cache": {3: attn_cache_len},
    }
    arg_inputs_cache = (
        torch.randn((2, 100, 80), dtype=dtype, device=device),
        torch.randn((2, 100, 80), dtype=dtype, device=device),
        torch.randn((2,), dtype=dtype, device=device),
        torch.randn((2, 100, 80), dtype=dtype, device=device),
        torch.randn((2, 100, 80), dtype=dtype, device=device),
        torch.tensor([100], dtype=torch.int64, device=device),
        torch.randn((2, 2, 1024, 30), dtype=dtype, device=device),
        torch.randn((22, 2, 16, 100, 64), dtype=dtype, device=device),
        torch.randn((22, 2, 16, 100, 64), dtype=dtype, device=device),
    )
    dynamic_shapes_no_cache = {"x": {1: seq_len}, "mu": {1: seq_len}, "t": {}, "spks": {1: seq_len}, "cond": {1: seq_len}}
    arg_inputs_no_cache = (
        torch.randn((2, 100, 80), dtype=dtype, device=device),
        torch.randn((2, 100, 80), dtype=dtype, device=device),
        torch.randn((2,), dtype=dtype, device=device),
        torch.randn((2, 100, 80), dtype=dtype, device=device),
        torch.randn((2, 100, 80), dtype=dtype, device=device),
    )
    return {
        "cache-in-cache-out": {"dynamic_shapes": dynamic_shapes_cache, "arg_inputs": arg_inputs_cache, "class": WrappedDiTCacheInCacheOut},
        "cache-in": {"dynamic_shapes": dynamic_shapes_cache, "arg_inputs": arg_inputs_cache, "class": WrappedDiTCacheIn},
        "cache-out": {"dynamic_shapes": dynamic_shapes_no_cache, "arg_inputs": arg_inputs_no_cache, "class": WrappedDiTCacheOut},
        "no-cache": {"dynamic_shapes": dynamic_shapes_no_cache, "arg_inputs": arg_inputs_no_cache, "class": WrappedDiT},
    }


def convert_entry(
    dit_model: DiT,
    compiled_types: List[Literal["cache-in-cache-out", "cache-in", "cache-out", "no-cache"]],
    output_dir: str,
    dtype=torch.float16,
    precisions={torch.float16},
    workspace_size=4 << 30,
    device="cuda",
):
    config_mapping = build_mapping(dtype, device)

    output_models = []
    for this_type in compiled_types:
        saved_path = os.path.join(output_dir, f"flow_stream_{this_type}_{str(dtype).split('.')[1]}.pt2")

        if not os.path.exists(saved_path):
            config = config_mapping[this_type]
            tgt_model = config["class"](dit_model)
            arg_inputs = config["arg_inputs"]
            dynamic_shapes = config["dynamic_shapes"]
            trt_model = export_and_compile(tgt_model, this_type, arg_inputs, dynamic_shapes, dtype, precisions, workspace_size, device)
            os.makedirs(output_dir, exist_ok=True)
            torch_tensorrt.save(trt_model, saved_path, arg_inputs=arg_inputs, output_format="exported_program")
            del trt_model
            gc.collect()
            torch.cuda.empty_cache()
            print(f"#### TRT model is successfully saved to: {saved_path}")
        else:
            print(f"#### Compiled model is existed: {saved_path}")

        loaded_model = torch_tensorrt.load(saved_path).module()
        output_models.append(loaded_model)

    return output_models


def convert_cv3_streamflow_to_trt(flow_model: CausalMaskedDiffWithDiT, version, output_dir, dtype, device, workspace_size):
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    workspace_size = int(workspace_size * (1024**3))
    flow_model.decoder.use_trt_estimator = True
    dit_model = flow_model.decoder.estimator

    convert_func = partial(
        convert_entry, dit_model=dit_model, output_dir=output_dir, dtype=dtype, precisions={dtype}, workspace_size=workspace_size, device=device
    )
    if version == "cosyvoice3_streamflow_cacheprompt":
        trt_model_ci, trt_model_co = convert_func(compiled_types=["cache-in", "cache-out"])
        flow_model.decoder.trt_estimator_cache_in = trt_model_ci
        flow_model.decoder.trt_estimator_cache_out = trt_model_co
    elif version == "cosyvoice3_streamflow_stream":
        trt_model_cico, trt_model_co = convert_func(compiled_types=["cache-in-cache-out", "cache-out"])
        flow_model.decoder.trt_estimator_cico = trt_model_cico
        flow_model.decoder.trt_estimator_cache_out = trt_model_co
    else:
        trt_model = convert_func(compiled_types=["no-cache"])[0]
        flow_model.decoder.trt_estimator = trt_model
