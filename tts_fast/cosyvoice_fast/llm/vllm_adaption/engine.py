import os
import shutil
import uuid

from .config import ENGINE_ARGS, SAMPLING_PARAMS
import vllm
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams


def register_model(model_name, forced):
    vllm_path = os.path.dirname(vllm.__file__)
    vllm_models_dir = os.path.join(vllm_path, "model_executor", "models")
    tgt_model_file = os.path.join(vllm_models_dir, f"{model_name}.py")

    if os.path.exists(tgt_model_file) and not forced:
        print(
            f"The model `{model_name}` is already registered. "
            "Use `forced=True` to overwrite."
        )
        return

    src_model_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", f"{model_name}.py"
    )
    shutil.copy(src_model_file, tgt_model_file)

    registry_path = os.path.join(vllm_models_dir, "registry.py")
    class_name = "CosyVoice2LLM" if model_name == "cosyvoice2llm" else "CosyVoice3LLM"

    with open(registry_path, "r") as f:
        lines = f.readlines()

    remove_pos = None
    for i, line in enumerate(lines):
        if class_name in line:
            remove_pos = i
            break
    if remove_pos is not None:
        if not forced:
            print(
                f"The model `{model_name}` is already registered. "
                "Use `forced=True` to overwrite."
            )
            return
        else:
            lines.pop(remove_pos)

    insert_pos = None
    for i, line in enumerate(lines):
        if line.strip().startswith("_VLLM_MODELS"):
            insert_pos = i + 1
            break
    if insert_pos is None:
        raise ValueError("Could not find insertion position!")
    lines.insert(
        insert_pos,
        f'    "{class_name}": ("{model_name}", "{class_name}"),\n',
    )
    with open(registry_path, "w") as f:
        f.writelines(lines)

    print(f"Successfully register model `{model_name}`!")


def get_generation_fn(model_dir, model_name="cosyvoice2llm", force_registration=True):
    model_name = model_name.lower()
    register_model(model_name, force_registration)

    ENGINE_ARGS["hf_overrides"] = (
        {"architectures": ["CosyVoice2LLM"]}
        if model_name == "cosyvoice2llm"
        else {"architectures": ["CosyVoice3LLM"]}
    )
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model=model_dir, **ENGINE_ARGS)
    )

    async def generation_fn(
        input_token_ids,
        stop_token_ids,
        max_tokens=None,
        min_tokens=0,
    ):
        sampling_params = SamplingParams(
            **SAMPLING_PARAMS,
            stop_token_ids=stop_token_ids,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )
        async for output in engine.generate(
            {"prompt_token_ids": input_token_ids},
            sampling_params=sampling_params,
            request_id=uuid.uuid4().hex,
        ):
            yield output.outputs[0].token_ids

    return generation_fn
