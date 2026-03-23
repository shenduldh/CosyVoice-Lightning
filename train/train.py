import os
import sys

__project_root__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(__project_root__, "tts_fast"))
sys.path.insert(0, os.path.join(__project_root__, "tts_fast/third_party/Matcha-TTS"))

from typing import Literal
import torch
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import torch.distributed as dist
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader
from copy import deepcopy
import json

from cosyvoice.dataset.dataset import Dataset
from cosyvoice.utils.losses import DPOLoss


def set_logger(saved_dir):
    if int(os.environ.get("RANK", 0)) == 0:
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        logger.add(
            os.path.join(saved_dir, "training.log"),
            format=logger_format,
            level="INFO",
        )
    else:
        logger.info = lambda *args: None


def load_config(
    config_path, model_name, qwen_pretrain_path, use_amp, use_deepspeed, ds_config_path
):
    overrides = {
        "llm": None,
        "flow": None,
        "hift": None,
        "hifigan": None,
        "qwen_pretrain_path": qwen_pretrain_path,
    }
    match model_name:
        case "llm":
            overrides.pop("llm")
        case "flow":
            overrides.pop("flow")
        case "hifigan":
            overrides.pop("hift")
            overrides.pop("hifigan")
    with open(config_path, "r") as f:
        config = load_hyperpyyaml(f, overrides=overrides)

    train_hifigan = model_name == "hifigan"
    training_config = (
        config["train_conf_gan"] if train_hifigan else config["train_conf"]
    )
    training_config["train_hifigan"] = train_hifigan

    if not use_deepspeed:
        training_config["dtype"] = torch.bfloat16 if use_amp else torch.float32
    else:
        with open(ds_config_path, "r") as f:
            ds_config = json.load(f)

        if "fp16" in ds_config and ds_config["fp16"]["enabled"]:
            training_config["dtype"] = torch.float16
        elif "bf16" in ds_config and ds_config["bf16"]["enabled"]:
            training_config["dtype"] = torch.bfloat16
        else:
            training_config["dtype"] = torch.float32

        training_config["accum_grad"] = ds_config["gradient_accumulation_steps"]
        training_config["grad_clip"] = ds_config["gradient_clipping"]

    return config, training_config


def load_model(config, model_name, checkpoint_path, apply_dpo, ref_checkpoint_path):
    model = config[model_name]

    curr_epoch, curr_step = 0, 0
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        if "epoch" in state_dict:
            curr_epoch = state_dict["epoch"]
        if "step" in state_dict:
            curr_step = state_dict["step"]

    if apply_dpo and model_name == "llm":
        model.forward = model.forward_dpo
        ref_model = deepcopy(model)
        if ref_checkpoint_path is not None:
            state_dict = torch.load(ref_checkpoint_path, map_location="cpu")
            ref_model.load_state_dict(state_dict, strict=False)
        ref_model.eval()
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)
    else:
        ref_model = None
        dpo_loss = None

    return model, ref_model, dpo_loss, curr_epoch, curr_step


def load_ref_model(config, model_name, checkpoint_path, apply_dpo, ref_checkpoint_path):
    model = config[model_name]

    curr_epoch, curr_step = 0, 0
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        if "epoch" in state_dict:
            curr_epoch = state_dict["epoch"]
        if "step" in state_dict:
            curr_step = state_dict["step"]

    if apply_dpo and model_name == "llm":
        model.forward = model.forward_dpo
        ref_model = deepcopy(model)
        if ref_checkpoint_path is not None:
            state_dict = torch.load(ref_checkpoint_path, map_location="cpu")
            ref_model.load_state_dict(state_dict, strict=False)
        ref_model.eval()
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)
    else:
        ref_model = None
        dpo_loss = None

    return model, ref_model, dpo_loss, curr_epoch, curr_step


def load_dataloaders(
    config,
    model_name,
    train_data_list_path,
    dev_data_list_path,
    pin_memory,
    num_workers,
    prefetch_factor,
    apply_dpo,
):
    data_pipeline = (
        config["data_pipeline_gan"]
        if model_name == "hifigan"
        else config["data_pipeline"]
    )
    train_dataset = Dataset(
        train_data_list_path,
        data_pipeline=data_pipeline,
        mode="train",
        gan=model_name == "hifigan",
        dpo=apply_dpo,
        shuffle=True,
        partition=True,
    )
    dev_dataset = Dataset(
        dev_data_list_path,
        data_pipeline=data_pipeline,
        mode="train",
        gan=model_name == "hifigan",
        dpo=apply_dpo,
        shuffle=False,
        partition=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=None,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    return train_dataloader, dev_dataloader


def eval_func(model, dev_dataloader, eval_hifigan):
    dist.barrier()
    cuda_id = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{cuda_id}")
    model.eval()
    total_num_utts, total_loss_dict = 0, {}
    for batch_idx, batch in tqdm(
        enumerate(dev_dataloader),
        desc="Eval",
        disable=dist.get_rank() != 0,
    ):
        if eval_hifigan:
            batch["turn"] = "generator"
        with torch.no_grad():
            loss_dict = model(batch, device)

        num_utts = len(batch["utts"])
        total_num_utts += num_utts

        for k, v in loss_dict.items():
            if k not in total_loss_dict:
                total_loss_dict[k] = []
            total_loss_dict[k].append(v.mean().item() * num_utts)

    for k, v in total_loss_dict.items():
        total_loss_dict[k] = sum(v) / total_num_utts

    return total_loss_dict


def main(
    model_name: Literal["llm", "flow", "hifigan"],
    config_path: str,
    train_data_list_path: str,
    dev_data_list_path: str,
    qwen_pretrain_path: str,
    checkpoint_path: str,
    apply_dpo: bool,
    ref_checkpoint_path: str,
    saved_dir: str,
    use_amp=True,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=100,
    num_epochs=200,
    eval_steps=-1,
    use_deepspeed=False,
    ds_config_path=None,
):
    if use_deepspeed:
        assert model_name in ["llm", "flow"]
        from deepspeed_utils import (
            init_distributed,
            wrap_model,
            load_optimizer_and_scheduler,
            save_model,
            train_one_epoch,
        )
    else:
        from ddp_utils import (
            init_distributed,
            wrap_model,
            load_optimizer_and_scheduler,
            save_model,
            train_one_epoch,
        )

    # get saving directory path
    saved_dir = os.path.join(
        saved_dir, model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    # init dsitribution
    init_distributed()
    # set logger
    set_logger(saved_dir)
    # load configs
    config, training_config = load_config(
        config_path,
        model_name,
        qwen_pretrain_path,
        use_amp,
        use_deepspeed,
        ds_config_path,
    )
    # load model
    model, ref_model, dpo_loss, init_epoch, init_step = load_model(
        config, model_name, checkpoint_path, apply_dpo, ref_checkpoint_path
    )
    training_config["epoch"] = init_epoch
    training_config["step"] = init_step
    # wrap model
    model = wrap_model(model, training_config)
    if ref_model is not None:
        ref_model = wrap_model(ref_model, training_config)
    # load optimizer and scheduler
    model, optimizer, scheduler, optimizer_disc, scheduler_disc = (
        load_optimizer_and_scheduler(model, training_config)
    )
    # load dataloaders
    train_dataloader, dev_dataloader = load_dataloaders(
        config,
        model_name,
        train_data_list_path,
        dev_data_list_path,
        pin_memory,
        num_workers,
        prefetch_factor,
        apply_dpo,
    )

    # save model once before training
    model_saved_dir = os.path.join(saved_dir, "checkpoints")
    save_model(model, training_config, model_saved_dir, "init")

    scaler = torch.amp.GradScaler() if use_amp and not use_deepspeed else None
    for epoch in range(init_epoch, num_epochs):
        logger.info(f"Epoch {epoch}")

        training_config["epoch"] = epoch

        # train an epoch
        train_one_epoch(
            model=model,
            ref_model=ref_model,
            train_dataloader=train_dataloader,
            dev_dataloader=dev_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_disc=optimizer_disc,
            scheduler_disc=scheduler_disc,
            dpo_loss=dpo_loss,
            scaler=scaler,
            training_config=training_config,
            eval_func=eval_func,
            eval_steps=eval_steps,
            saved_dir=model_saved_dir,
        )

        # eval when an epoch ends
        loss_dict = eval_func(model, dev_dataloader, training_config["train_hifigan"])
        logger.info(f"Epoch End -- loss_dict: {loss_dict}")

        # save model when an epoch ends
        save_model(model, training_config, model_saved_dir, "epochend", loss_dict)

    dist.destroy_process_group()


if __name__ == "__main__":
    main(
        model_name="llm",
        config_path="./config.yaml",
        train_data_list_path="./output/parquet/parquet_list",
        dev_data_list_path="./output/parquet/parquet_list",
        qwen_pretrain_path="../assets/CosyVoice2-0.5B/CosyVoice-BlankEN",
        checkpoint_path="../assets/CosyVoice2-0.5B/llm.pt",
        apply_dpo=True,
        ref_checkpoint_path="../assets/CosyVoice2-0.5B/llm.pt",
        saved_dir="./saved",
        use_amp=True,
        num_workers=0,
        pin_memory=True,
        prefetch_factor=None,
        num_epochs=10,
        eval_steps=200,
        use_deepspeed=False,
        ds_config_path="./ds_config.json",
    )
