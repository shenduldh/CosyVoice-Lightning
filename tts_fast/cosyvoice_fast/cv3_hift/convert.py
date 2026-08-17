import os
import gc
import torch
from torch import nn
import torch_tensorrt

from .model import CausalHiFTGenerator, CausalConvRNNF0Predictor


class HiFTFinal(nn.Module):
    def __init__(self, hift_model: CausalHiFTGenerator):
        super().__init__()
        self.hift_model = hift_model

    def forward(self, x):
        y = self.hift_model.inference(x, finalize=True)
        return y


class HiFTNotFinal(nn.Module):
    def __init__(self, hift_model: CausalHiFTGenerator):
        super().__init__()
        self.hift_model = hift_model

    def forward(self, x):
        y = self.hift_model.inference(x, finalize=False)
        return y


class WrappedModel(nn.Module):
    def __init__(self, trt_hifi_final, trt_hift_notfinal):
        super().__init__()
        self.trt_hifi_final = trt_hifi_final
        self.trt_hift_notfinal = trt_hift_notfinal

    @torch.inference_mode()
    def inference(self, speech_feat, finalize):
        y = self.trt_hifi_final(speech_feat) if finalize else self.trt_hift_notfinal(speech_feat)
        return y


def export_and_compile(model, saved_path, arg_inputs, dynamic_shapes, dtype, precisions, workspace_size, device):
    model.eval().to(device, dtype)

    ep = torch.export.export(model, args=arg_inputs, dynamic_shapes=dynamic_shapes, strict=False)
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

    os.makedirs(os.path.dirname(saved_path), exist_ok=True)
    torch_tensorrt.save(trt_model, saved_path, inputs=arg_inputs, output_format="torchscript")
    print(f"#### TRT model is successfully saved to: {saved_path}")

    del trt_model
    gc.collect()
    torch.cuda.empty_cache()


def convert_hift_to_trt(
    hift_model: CausalHiFTGenerator,
    output_dir: str,
    dtype=torch.float32,
    precisions={torch.float32},
    workspace_size=1 << 30,
    device="cuda",
):
    saved_path_final = os.path.join(output_dir, f"hift_{str(dtype).split('.')[1]}_final.ts")
    saved_path_notfinal = os.path.join(output_dir, f"hift_{str(dtype).split('.')[1]}_notfinal.ts")

    seq_len = torch.export.Dim("seq_len", min=16, max=3000)
    dynamic_shapes = {"x": {2: seq_len}}
    arg_inputs = (torch.randn((1, 80, 200), dtype=dtype, device=device),)

    if not os.path.exists(saved_path_final):
        model_final = HiFTFinal(hift_model)
        export_and_compile(model_final, saved_path_final, arg_inputs, dynamic_shapes, dtype, precisions, workspace_size, device)
    else:
        print(f"#### Compiled model is existed: {saved_path_final}")

    if not os.path.exists(saved_path_notfinal):
        model_notfinal = HiFTNotFinal(hift_model)
        export_and_compile(model_notfinal, saved_path_notfinal, arg_inputs, dynamic_shapes, dtype, precisions, workspace_size, device)
    else:
        print(f"#### Compiled model is existed: {saved_path_notfinal}")

    loaded_model_final = torch_tensorrt.load(saved_path_final)
    loaded_model_notfinal = torch_tensorrt.load(saved_path_notfinal)

    wrapped = WrappedModel(loaded_model_final, loaded_model_notfinal)
    return wrapped


if __name__ == "__main__":
    import time

    hift_model = CausalHiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        conv_pre_look_right=4,
        f0_predictor=CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512),
    )
    hift_model.remove_parametrizations()
    trt_model = convert_hift_to_trt(hift_model, "./test")
    hift_model.to("cuda").eval()

    dummy_input = torch.randn((1, 80, 200), device="cuda")
    for _ in range(10):
        s = time.perf_counter()
        hift_model.inference(dummy_input, False)
        e = time.perf_counter()
        print(e - s)
    for _ in range(10):
        s = time.perf_counter()
        trt_model.inference(dummy_input, False)
        e = time.perf_counter()
        print(e - s)

    output1 = hift_model.inference(dummy_input, False)
    output2 = trt_model.inference(dummy_input, False)
    print((output1 - output2).abs().sum().item())

    output1 = hift_model.inference(dummy_input, True)
    output2 = trt_model.inference(dummy_input, True)
    print((output1 - output2).abs().sum().item())
