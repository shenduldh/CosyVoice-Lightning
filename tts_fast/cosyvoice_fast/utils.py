import os
import torch
import types
import queue
import re
from loguru import logger
import threading
import time
from contextlib import contextmanager
import gc


class EstimatorPool:
    def __init__(self, estimator_engine, estimator_count, device):
        estimator_count = max(estimator_count, 1)
        self.pool = queue.Queue(maxsize=estimator_count)
        self.engine = estimator_engine
        for _ in range(estimator_count):
            estimator = self.engine.create_execution_context()
            stream = torch.cuda.Stream(device)
            assert estimator is not None
            self.pool.put((estimator, stream))
        assert not self.pool.empty()

    def get(self):
        estimator, stream = self.pool.get()
        return estimator, stream, self.engine

    def put(self, estimator, stream):
        self.pool.put((estimator, stream))


def forward_estimator(self, x, mask, mu, t, spks, cond, *args, **kwargs):
    estimator, executed_stream, engine = self.estimator_pool.get()

    try:
        producer_stream = torch.cuda.current_stream()
        with torch.cuda.stream(executed_stream):
            executed_stream.wait_stream(producer_stream)

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

            assert estimator.execute_async_v3(executed_stream.cuda_stream)

        producer_stream.wait_stream(executed_stream)
        return x

    finally:
        self.estimator_pool.put(estimator, executed_stream)


def get_data_ptr(tensor: torch.Tensor, dummy_buffer: torch.Tensor):
    if tensor.numel() == 0:
        return dummy_buffer.data_ptr()
    else:
        return tensor.contiguous().data_ptr()


def forward_estimator_chunk(self, x, mu, t, spks, cond, cnn_cache, att_cache):
    estimator, executed_stream, engine = self.estimator_pool.get()

    try:
        producer_stream = torch.cuda.current_stream()
        with torch.cuda.stream(executed_stream):
            executed_stream.wait_stream(producer_stream)

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

        producer_stream.wait_stream(executed_stream)
        return x, new_cnn_cache, new_att_cache

    finally:
        self.estimator_pool.put(estimator, executed_stream)


def set_flow_decoder_estimator(flow, estimator_engine, device, estimator_count=1):
    del flow.decoder.estimator
    flow.decoder.estimator_pool = EstimatorPool(estimator_engine, estimator_count, device)
    flow.decoder.forward_estimator = types.MethodType(forward_estimator, flow.decoder)
    flow.decoder.forward_estimator_chunk = types.MethodType(forward_estimator_chunk, flow.decoder)


def get_flow_decoder_estimator_input_shapes():
    min_btz, opt_btz, max_btz = 2 * 1, 2 * 1, 2 * 1
    min_shapes = [(min_btz, 80, 4), (min_btz, 1, 4), (min_btz, 80, 4), (min_btz,), (min_btz, 80), (min_btz, 80, 4)]
    opt_shapes = [(opt_btz, 80, 193), (opt_btz, 1, 193), (opt_btz, 80, 193), (opt_btz,), (opt_btz, 80), (opt_btz, 80, 193)]
    max_shapes = [(max_btz, 80, 6800), (max_btz, 1, 6800), (max_btz, 80, 6800), (max_btz,), (max_btz, 80), (max_btz, 80, 6800)]
    input_names = ["x", "mask", "mu", "t", "spks", "cond"]
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
        match = re.search(r"ir_version [0-9]+ is higher than the checker's \(([0-9]+)\)", str(e))
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
    workspace_size=4,
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
    elif dtype == "fp32":
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


class CudaCacheCleaner:
    def __init__(
        self,
        delay: float = 2.0,  # 防抖延迟（秒）
        auto_clean_interval: float | None = 300.0,  # 自动定期清理间隔
        min_reclaimable_gb: float = 0.0,  # 触发清理的可回收显存阈值 (GB)
        device: int | str | torch.device = "cuda",
        enabled: bool = True,
    ):
        self.delay = delay
        self.auto_clean_interval = auto_clean_interval
        self.min_reclaimable_bytes = int(min_reclaimable_gb * (1024**3))
        self.device = torch.device(device) if isinstance(device, (int, str)) else device
        self.enabled = enabled

        self.lock = threading.Lock()
        self._executing_count = 0

        self.timer: threading.Timer | None = None
        self._auto_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        if self.auto_clean_interval and self.auto_clean_interval > 0:
            self.start_auto_cleanup(self.auto_clean_interval)

    @contextmanager
    def track(self):
        self._executing_count += 1
        try:
            yield
        finally:
            self._executing_count = max(0, self._executing_count - 1)

    def empty_cache(self):
        if self.enabled:
            with self.lock:
                self._cancel_timer()
                timer = threading.Timer(self.delay, self._execute_cleanup)
                timer.daemon = True
                self.timer = timer
                timer.start()

    def _cancel_timer(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def _execute_cleanup(self):
        if not torch.cuda.is_available():
            return

        if self._executing_count > 0:
            return

        reserved = torch.cuda.memory_reserved(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        if (reserved - allocated) < self.min_reclaimable_bytes:
            return

        gc.collect()
        torch.cuda.empty_cache()

    def start_auto_cleanup(self, interval: float):
        if not self.enabled:
            return

        self.stop_auto_cleanup()
        self.auto_clean_interval = interval
        self._stop_event.clear()

        def _loop():
            while not self._stop_event.wait(timeout=self.auto_clean_interval):
                self.empty_cache()

        self._auto_thread = threading.Thread(target=_loop, daemon=True)
        self._auto_thread.start()

    def stop_auto_cleanup(self):
        self._stop_event.set()
        if self._auto_thread:
            timeout = (self.auto_clean_interval or 1.0) + 3.0
            self._auto_thread.join(timeout=timeout)
        self._auto_thread = None

    def stop(self):
        self.stop_auto_cleanup()
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
