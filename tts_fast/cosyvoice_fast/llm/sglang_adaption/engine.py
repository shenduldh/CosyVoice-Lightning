import os
import shutil
import sglang
from .config import ENGINE_ARGS, SAMPLING_PARAMS


def register_model(model_name, forced):
    sglang_path = os.path.dirname(sglang.__file__)
    sglang_models_dir = os.path.join(sglang_path, "srt", "models")
    tgt_model_file = os.path.join(sglang_models_dir, f"{model_name}.py")

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
    print(f"Successfully register model `{model_name}`!")


def get_generation_fn(model_dir, model_name="cosyvoice2llm", force_registration=True):
    model_name = model_name.lower()
    register_model(model_name, force_registration)
    ENGINE_ARGS["json_model_override_args"] = (
        '{"architectures": ["CosyVoice2LLM"], "vocab_size": 6564}'
        if model_name == "cosyvoice2llm"
        else '{"architectures": ["CosyVoice3LLM"], "vocab_size": 6761}'
    )
    engine = sglang.Engine(model_path=model_dir, **ENGINE_ARGS)

    async def generation_fn(
        input_token_ids, stop_token_ids, max_tokens=None, min_tokens=0
    ):
        sampling_params = {
            **SAMPLING_PARAMS,
            "stop_token_ids": stop_token_ids,
            "max_new_tokens": max_tokens,
            "min_new_tokens": 0,
        }
        generator = await engine.async_generate(
            input_ids=input_token_ids, sampling_params=sampling_params, stream=True
        )
        last_completion_tokens = 0
        async for output in generator:
            yield output["output_ids"][last_completion_tokens:]
            last_completion_tokens = output["meta_info"]["completion_tokens"]

    return generation_fn
