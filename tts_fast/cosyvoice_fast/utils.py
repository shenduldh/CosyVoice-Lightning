import os
import torch
import types
from typing import Literal
import queue
import re
from loguru import logger


class EstimatorPool:
    def __init__(self, estimator_engine, estimator_count, device):
        estimator_count = max(estimator_count, 1)
        self.pool = queue.Queue(maxsize=estimator_count)
        self.engine = estimator_engine
        for _ in range(estimator_count):
            estimator = self.engine.create_execution_context()
            stream = torch.cuda.stream(torch.cuda.Stream(device))
            assert estimator is not None
            self.pool.put((estimator, stream))
        assert not self.pool.empty()

    def get(self):
        estimator, context = self.pool.get()
        torch.cuda.current_stream().synchronize()
        return estimator, context, self.engine

    def put(self, estimator, context):
        torch.cuda.current_stream().synchronize()
        self.pool.put((estimator, context))


def forward_estimator(self, x, mask, mu, t, spks, cond, *args, **kwargs):
    estimator, context, engine = self.estimator_pool.get()
    with context:
        btz, _, in_dims = x.size()
        estimator.set_input_shape("x", (btz, 80, in_dims))
        estimator.set_input_shape("mask", (btz, 1, in_dims))
        estimator.set_input_shape("mu", (btz, 80, in_dims))
        estimator.set_input_shape("t", (btz,))
        estimator.set_input_shape("spks", (btz, 80))
        estimator.set_input_shape("cond", (btz, 80, in_dims))

        data_ptrs = [
            x.contiguous().data_ptr(),
            mask.contiguous().data_ptr(),
            mu.contiguous().data_ptr(),
            t.contiguous().data_ptr(),
            spks.contiguous().data_ptr(),
            cond.contiguous().data_ptr(),
            x.data_ptr(),
        ]

        for idx, data_ptr in enumerate(data_ptrs):
            estimator.set_tensor_address(engine.get_tensor_name(idx), data_ptr)

        assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        self.estimator_pool.put(estimator, context)
        return x


def get_data_ptr(tensor: torch.Tensor, dummy_buffer: torch.Tensor):
    if tensor.numel() == 0:
        return dummy_buffer.data_ptr()
    else:
        return tensor.contiguous().data_ptr()


def forward_estimator_chunk(self, x, mu, t, spks, cond, cnn_cache, att_cache):
    estimator, context, engine = self.estimator_pool.get()
    with context:
        btz, _, in_dims = x.size()
        # att_cache = att_cache[:, :, :, : 1000 - in_dims, :]

        estimator.set_input_shape("x", (btz, 80, in_dims))
        estimator.set_input_shape("mu", (btz, 80, in_dims))
        estimator.set_input_shape("t", (btz,))
        estimator.set_input_shape("spks", (btz, 80))
        estimator.set_input_shape("cond", (btz, 80, in_dims))
        estimator.set_input_shape("cnn_cache", cnn_cache.shape)
        estimator.set_input_shape("att_cache", att_cache.shape)

        new_cnn_cache = torch.empty_like(cnn_cache)
        new_att_cache_shape = list(att_cache.shape)
        new_att_cache_shape[3] += in_dims
        new_att_cache = torch.empty(new_att_cache_shape, device=att_cache.device, dtype=x.dtype)

        data_ptrs = [
            x.contiguous().data_ptr(),
            mu.contiguous().data_ptr(),
            t.contiguous().data_ptr(),
            spks.contiguous().data_ptr(),
            cond.contiguous().data_ptr(),
            cnn_cache.contiguous().data_ptr(),
            get_data_ptr(att_cache, self.dummy_buffer),
            x.data_ptr(),
            new_cnn_cache.data_ptr(),
            get_data_ptr(new_att_cache, self.dummy_buffer),
        ]

        for i, j in enumerate(data_ptrs):
            estimator.set_tensor_address(engine.get_tensor_name(i), j)

        assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        self.estimator_pool.put(estimator, context)

        return x, new_cnn_cache, new_att_cache


def set_flow_decoder_estimator(flow, estimator_engine, device, estimator_count=1):
    del flow.decoder.estimator
    flow.decoder.estimator_pool = EstimatorPool(estimator_engine, estimator_count, device)
    flow.decoder.forward_estimator = types.MethodType(forward_estimator, flow.decoder)
    flow.decoder.forward_estimator_chunk = types.MethodType(forward_estimator_chunk, flow.decoder)


def get_flow_decoder_estimator_input_shapes(
    flow_type: Literal["cosyvoice2", "cosyvoice2_stepaudio_stream", "cosyvoice2_stepaudio_whole", "cosyvoice3"] = "cosyvoice2",
):
    match flow_type:
        case "cosyvoice2" | "cosyvoice3":
            min_shapes = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2,), (2, 80), (2, 80, 4)]
            opt_shapes = [(2, 80, 193), (2, 1, 193), (2, 80, 193), (2,), (2, 80), (2, 80, 193)]
            max_shapes = [(2, 80, 6800), (2, 1, 6800), (2, 80, 6800), (2,), (2, 80), (2, 80, 6800)]
            input_names = ["x", "mask", "mu", "t", "spks", "cond"]
        case "cosyvoice2_stepaudio_stream":
            opt_btz = max_btz = 1 * 2
            min_shapes = [(2, 80, 4), (2, 80, 4), (2, 80, 4), (2,), (2, 80), (16, 2, 1024, 2), (16, 2, 8, 0, 128)]
            opt_shapes = [
                (opt_btz, 80, 500),
                (opt_btz, 80, 500),
                (opt_btz, 80, 500),
                (opt_btz,),
                (opt_btz, 80),
                (16, opt_btz, 1024, 2),
                (16, opt_btz, 8, 100, 128),
            ]
            max_shapes = [
                (max_btz, 80, 3000),
                (max_btz, 80, 3000),
                (max_btz, 80, 3000),
                (max_btz,),
                (max_btz, 80),
                (16, max_btz, 1024, 2),
                (16, max_btz, 8, 1000, 128),
            ]
            input_names = ["x", "mu", "cond", "t", "spks", "cnn_cache", "att_cache"]
        case "cosyvoice2_stepaudio_whole":
            opt_btz, max_btz = 2 * 2, 2 * 16
            min_shapes = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4), (2,), (2, 80)]
            opt_shapes = [
                (opt_btz, 80, 500),
                (opt_btz, 1, 500),
                (opt_btz, 80, 500),
                (opt_btz, 80, 500),
                (opt_btz,),
                (opt_btz, 80),
            ]
            max_shapes = [
                (max_btz, 80, 3000),
                (max_btz, 1, 3000),
                (max_btz, 80, 3000),
                (max_btz, 80, 3000),
                (max_btz,),
                (max_btz, 80),
            ]
            input_names = ["x", "mask", "mu", "cond", "t", "spks"]
    return zip(input_names, min_shapes, opt_shapes, max_shapes)


def simplify_onnx(onnx_path):
    import onnx
    import onnxsim

    logger.info(f"Simplify {onnx_path}.")

    orig_path = onnx_path
    simplified_path = onnx_path.replace(".onnx", ".simplified.onnx")

    if os.path.exists(simplified_path):
        logger.info(f"Simplified model {simplified_path} is existed.")
        return simplified_path

    orig_model = onnx.load(orig_path)

    try:
        simplified_model, check = onnxsim.simplify(orig_model)
    except Exception as e:
        match = re.search("ir_version [0-9]+ is higher than the checker's \(([0-9]+)\)", str(e))
        if match:
            ir_version = int(match.group(1))
            logger.info(f"Downgrade `ir_version` to {ir_version} and continue simplification.")
            orig_model.ir_version = ir_version
            simplified_model, check = onnxsim.simplify(orig_model)
        else:
            logger.error(f"Simplification failed due to {e}.")
            return orig_path

    if check:
        onnx.save(simplified_model, simplified_path)
        logger.info(f"Simplify successfully to {simplified_path}.")
        return simplified_path

    logger.error("Simplification failed due to invalid check.")
    return orig_path


def slim_onnx(onnx_path):
    import onnx
    import onnxslim

    logger.info(f"Slim {onnx_path}.")

    orig_path = onnx_path
    slimmed_path = onnx_path.replace(".onnx", ".slimmed.onnx")

    if os.path.exists(slimmed_path):
        logger.info(f"Slimmed model {slimmed_path} is existed.")
        return slimmed_path

    orig_model = onnx.load(orig_path)
    slimmed_model = onnxslim.slim(orig_model)
    if slimmed_model:
        onnx.save(slimmed_model, slimmed_path)
        logger.info(f"Slim successfully to {slimmed_path}.")
        return slimmed_path

    logger.error("Slimming failed.")
    return orig_path


def export_flow_decoder_estimator_onnx(flow_model, onnx_saved_path, device):
    estimator = flow_model.decoder.estimator
    estimator.to(device).eval()

    x = torch.rand((2, 80, 256)).float().to(device)
    mask = torch.ones((2, 1, 256)).float().to(device)
    mu = torch.rand((2, 80, 256)).float().to(device)
    t = torch.rand((2)).float().to(device)
    spks = torch.rand((2, 80)).float().to(device)
    cond = torch.rand((2, 80, 256)).float().to(device)

    torch.onnx.export(
        estimator,
        (x, mask, mu, t, spks, cond),
        onnx_saved_path,
        export_params=True,
        opset_version=20,
        input_names=["x", "mask", "mu", "t", "spks", "cond"],
        output_names=["estimator_out"],
        dynamic_axes={
            "x": {2: "seq_len"},
            "mask": {2: "seq_len"},
            "mu": {2: "seq_len"},
            "cond": {2: "seq_len"},
            "estimator_out": {2: "seq_len"},
        },
    )


def autocast_onnx(onnx_path, dtype, calibration_data, **kwargs):
    import onnx
    from modelopt.onnx.autocast import convert_to_mixed_precision

    logger.info(f"Autocast {onnx_path} to {dtype}.")

    orig_path = onnx_path
    autocast_path = onnx_path.replace(".onnx", f".autocast_{dtype}.onnx")

    if os.path.exists(autocast_path):
        logger.info(f"Autocast model {autocast_path} is existed.")
        return autocast_path

    converted = convert_to_mixed_precision(
        onnx_path=orig_path,
        low_precision_type=dtype,
        data_max=512,
        calibration_data=calibration_data,
        providers=["cuda"],
        opset=20,
        init_max=65504,
        **kwargs,
    )
    onnx.save(converted, autocast_path)

    logger.info(f"Autocast successfully to {autocast_path}.")

    return autocast_path


def convert_onnx_to_trt(
    onnx_path: str,
    trt_path: str,
    input_shapes,
    dtype: str,
    workspace_size=8,
    optimization_level=3,
    timing_cache_path=None,
    set_layer_precision=None,
):
    import tensorrt as trt
    import onnx

    logger.info(f"Convert {onnx_path} to {trt_path}.")

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flag)
    parser = trt.OnnxParser(network, trt_logger)
    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_size * (1024**3)))
    config.builder_optimization_level = optimization_level

    # load timing cache
    if timing_cache_path is not None:
        logger.info(f"Use timing cache: {timing_cache_path}.")
        buffer = b""
        if os.path.exists(timing_cache_path):
            with open(timing_cache_path, "rb") as f:
                buffer = f.read()
        timing_cache = config.create_timing_cache(buffer)
        config.set_timing_cache(timing_cache, ignore_mismatch=False)

    # parse onnx model
    onnx_model = onnx.load(onnx_path)
    if not parser.parse(onnx_model.SerializeToString()):
        for error in range(parser.num_errors):
            logger.error(parser.get_error(error))
        raise ValueError(f"Failed to parse {onnx_path}.")

    # set allowed precision
    # config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    if dtype == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif dtype == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    elif dtype == "tf32":
        config.set_flag(trt.BuilderFlag.TF32)

    # set layer precision
    if set_layer_precision is not None:
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            layer_dtype = layer.get_output(0).dtype
            if layer_dtype in [trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16]:
                set_layer_precision(layer)

    # set input/output precision
    tensor_dtype = {"fp16": trt.DataType.HALF, "fp32": trt.DataType.FLOAT, "bf16": trt.DataType.BF16}[dtype]
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype

    # create profile and set input shapes
    profile = builder.create_optimization_profile()
    for name, min_shape, opt_shape, max_shape in input_shapes:
        profile.set_shape(name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # build trt engine
    plan = builder.build_serialized_network(network, config)

    # save trt engine
    if plan is None or plan.nbytes == 0:
        raise ValueError(f"Failed to build {trt_path}.")
    with open(trt_path, "wb") as f:
        f.write(plan)

    # save timing cache
    if timing_cache_path is not None:
        updated_cache = config.get_timing_cache()
        with open(timing_cache_path, "wb") as f:
            f.write(updated_cache.serialize())

    logger.info(f"Succesfully convert [{plan.nbytes / (1024**2):.2f} MB].")
