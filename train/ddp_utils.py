import sys

sys.path.insert(0, "/data1/other/cosyvoice2-lightning/tts_fast")
sys.path.insert(0, "/data1/other/cosyvoice2-lightning/tts_fast/third_party/Matcha-TTS")

import os
import torch
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch.nn.utils import clip_grad_norm_
from contextlib import nullcontext
import datetime
from loguru import logger
from tqdm import tqdm
import re
import json

from cosyvoice.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def wrap_model(model, training_config):
    model.cuda()
    model = DistributedDataParallel(
        model, find_unused_parameters=training_config["train_hifigan"]
    )
    return model


def load_optimizer_and_scheduler(model, training_config):
    if training_config["train_hifigan"]:
        # optimizer
        if training_config["optim"] == "adam":
            optimizer = optim.Adam(
                model.module.generator.parameters(), **training_config["optim_conf"]
            )
        elif training_config["optim"] == "adamw":
            optimizer = optim.AdamW(
                model.module.generator.parameters(), **training_config["optim_conf"]
            )
        # scheduler
        if training_config["scheduler"] == "warmuplr":
            scheduler = WarmupLR(optimizer, **training_config["scheduler_conf"])
        elif training_config["scheduler"] == "NoamHoldAnnealing":
            scheduler = NoamHoldAnnealing(
                optimizer, **training_config["scheduler_conf"]
            )
        elif training_config["scheduler"] == "constantlr":
            scheduler = ConstantLR(optimizer)
        # optimizer_disc
        if training_config["optim"] == "adam":
            optimizer_disc = optim.Adam(
                model.module.discriminator.parameters(), **training_config["optim_conf"]
            )
        elif training_config["optim"] == "adamw":
            optimizer_disc = optim.AdamW(
                model.module.discriminator.parameters(), **training_config["optim_conf"]
            )
        # scheduler_disc
        if training_config["scheduler"] == "warmuplr":
            scheduler_disc = WarmupLR(
                optimizer_disc, **training_config["scheduler_conf"]
            )
        elif training_config["scheduler"] == "NoamHoldAnnealing":
            scheduler_disc = NoamHoldAnnealing(
                optimizer_disc, **training_config["scheduler_conf"]
            )
        elif training_config["scheduler"] == "constantlr":
            scheduler_disc = ConstantLR(optimizer_disc)

    else:
        # optimizer
        if training_config["optim"] == "adam":
            optimizer = optim.Adam(model.parameters(), **training_config["optim_conf"])
        elif training_config["optim"] == "adamw":
            optimizer = optim.AdamW(model.parameters(), **training_config["optim_conf"])
        # scheduler
        if training_config["scheduler"] == "warmuplr":
            scheduler = WarmupLR(optimizer, **training_config["scheduler_conf"])
        elif training_config["scheduler"] == "NoamHoldAnnealing":
            scheduler = NoamHoldAnnealing(
                optimizer, **training_config["scheduler_conf"]
            )
        elif training_config["scheduler"] == "constantlr":
            scheduler = ConstantLR(optimizer)

        optimizer_disc, scheduler_disc = None, None

    curr_step = training_config["step"]
    scheduler.set_step(curr_step)
    if scheduler_disc is not None:
        scheduler_disc.set_step(curr_step)

    return model, optimizer, scheduler, optimizer_disc, scheduler_disc


def save_model(model, training_config, saved_dir, prefix, info_dict=None):
    if int(os.environ.get("RANK", 0)) == 0:
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        epoch = training_config["epoch"]
        step = training_config["step"]
        saved_path = os.path.join(saved_dir, f"{prefix}_{epoch}_{step}.pt")
        torch.save(
            {**model.module.state_dict(), "epoch": epoch, "step": step}, saved_path
        )

        if info_dict is not None:
            saved_path = re.sub(".pt$", ".json", saved_path)
            with open(saved_path, "w", encoding="utf-8") as f:
                json.dump(info_dict, f, ensure_ascii=False)


def forward(model, ref_model, batch, dpo_loss, scaler, dtype, device):
    with torch.amp.autocast("cuda", dtype=dtype, enabled=scaler is not None):
        loss_dict = model(batch, device)
        if ref_model is not None:
            with torch.no_grad():
                ref_loss_dict = ref_model(batch, device)
            chosen_logps = loss_dict["chosen_logps"]
            rejected_logps = loss_dict["rejected_logps"]
            ref_chosen_logps = ref_loss_dict["chosen_logps"]
            ref_rejected_logps = ref_loss_dict["rejected_logps"]
            preference_loss, chosen_reward, reject_reward = dpo_loss(
                chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
            )
            loss_dict["loss"] = preference_loss + loss_dict["loss"]
        return loss_dict["loss"]


def backward(loss, scaler):
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()


def update_params(model, scaler, optimizer, scheduler, grad_max_norm):
    if scaler is not None:
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), grad_max_norm)
        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
        scaler.update()
    else:
        grad_norm = clip_grad_norm_(model.parameters(), grad_max_norm)
        if torch.isfinite(grad_norm):
            optimizer.step()
    optimizer.zero_grad()
    scheduler.step()


def train_one_epoch(
    model,
    ref_model,
    train_dataloader,
    dev_dataloader,
    optimizer,
    scheduler,
    optimizer_disc,
    scheduler_disc,
    dpo_loss,
    scaler,
    training_config,
    eval_func,
    eval_steps,
    saved_dir=None,
    timeout=60,
):
    epoch = training_config["epoch"]
    train_hifigan = training_config["train_hifigan"]
    accum_grad_steps = 1 if train_hifigan else training_config["accum_grad"]
    grad_max_norm = training_config["grad_clip"]
    dtype = training_config["dtype"]

    dist.barrier()
    cuda_id = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{cuda_id}")
    dist_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=60))
    train_dataloader.dataset.set_epoch(epoch)

    with model.join():
        for batch_idx, batch in tqdm(
            enumerate(train_dataloader), desc="Train", disable=dist.get_rank() != 0
        ):
            model.train()

            if batch_idx > 0:
                try:
                    dist.monitored_barrier(
                        group=dist_group, timeout=datetime.timedelta(seconds=timeout)
                    )
                except:
                    break

            if not train_hifigan:
                # 是否结束累积梯度并更新参数
                do_step = (training_config["step"] + 1) % accum_grad_steps == 0
                # 梯度累积结束再在 cuda 间同步梯度
                context = nullcontext if do_step else model.no_sync
                with context():
                    loss = forward(
                        model, ref_model, batch, dpo_loss, scaler, dtype, device
                    )
                    backward(loss / accum_grad_steps, scaler)
                if do_step:
                    update_params(model, scaler, optimizer, scheduler, grad_max_norm)
                    training_config["step"] += 1
            else:
                batch["turn"] = "discriminator"
                loss = forward(model, ref_model, batch, dpo_loss, scaler, dtype, device)
                backward(loss / accum_grad_steps, scaler)
                update_params(
                    model, scaler, optimizer_disc, scheduler_disc, grad_max_norm
                )
                optimizer.zero_grad()
                batch["turn"] = "generator"
                loss = forward(model, ref_model, batch, dpo_loss, scaler, dtype, device)
                backward(loss / accum_grad_steps, scaler)
                update_params(model, scaler, optimizer, scheduler, grad_max_norm)
                optimizer_disc.zero_grad()
                training_config["step"] += 1

            if (
                eval_steps > 0
                and (training_config["step"] + 1) % eval_steps == 0
                and (batch_idx + 1) % accum_grad_steps == 0
            ):
                loss_dict = eval_func(model, dev_dataloader, train_hifigan)
                logger.info(f"Training -- loss_dict: {loss_dict}")
                if saved_dir is not None:
                    save_model(model, training_config, saved_dir, "training", loss_dict)

    dist.destroy_process_group(dist_group)
