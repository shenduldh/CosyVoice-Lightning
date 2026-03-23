import os
import torch
import torch.distributed as dist
import datetime
import argparse
from loguru import logger
from tqdm import tqdm
import deepspeed
from deepspeed.runtime.zero.stage_1_and_2 import (
    estimate_zero2_model_states_mem_needs_all_live,
)
import json

from cosyvoice.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR


def init_distributed():
    deepspeed.init_distributed(dist_backend="nccl")


def wrap_model(model, *args):
    if int(os.environ.get("RANK", 0)) == 0:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        estimate_zero2_model_states_mem_needs_all_live(
            model,
            num_gpus_per_node=local_world_size,
            num_nodes=world_size // local_world_size,
        )
    cuda_id = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{cuda_id}")
    model.to(device)
    return model


def load_optimizer_and_scheduler(model, training_config):
    def get_scheduler(opt):
        if training_config["scheduler"] == "warmuplr":
            scheduler = WarmupLR(opt, **training_config["scheduler_conf"])
        elif training_config["scheduler"] == "NoamHoldAnnealing":
            scheduler = NoamHoldAnnealing(opt, **training_config["scheduler_conf"])
        elif training_config["scheduler"] == "constantlr":
            scheduler = ConstantLR(opt)
        return scheduler

    parser = deepspeed.add_config_arguments(argparse.ArgumentParser())
    args = parser.parse_known_args()[0]
    model, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        lr_scheduler=get_scheduler,
        model_parameters=model.parameters(),
    )
    curr_step = training_config["step"]
    scheduler.set_step(curr_step)

    return model, optimizer, scheduler, None, None


def save_model(model, training_config, saved_dir, prefix, info_dict=None):
    with torch.no_grad():
        epoch = training_config["epoch"]
        step = training_config["step"]
        saved_name = f"{prefix}_{epoch}_{step}"
        model.save_checkpoint(
            save_dir=saved_dir,
            tag=saved_name,
            client_state={"epoch": epoch, "step": step},
        )

        if int(os.environ.get("RANK", 0)) == 0 and info_dict is not None:
            saved_path = os.path.join(saved_dir, f"{saved_name}.json")
            with open(saved_path, "w", encoding="utf-8") as f:
                json.dump(info_dict, f, ensure_ascii=False)


def forward(model, ref_model, batch, dpo_loss, dtype, device):
    with torch.amp.autocast("cuda", dtype=dtype, cache_enabled=False):
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


def backward(loss, model):
    model.backward(loss)


def update_params(model):
    model.step()


def train_one_epoch(
    model,
    ref_model,
    train_dataloader,
    dev_dataloader,
    scheduler,
    dpo_loss,
    training_config,
    eval_func,
    eval_steps,
    saved_dir=None,
    timeout=60,
    **kwargs,
):
    epoch = training_config["epoch"]
    dtype = training_config["dtype"]

    dist.barrier()
    cuda_id = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{cuda_id}")
    dist_group = dist.new_group(
        backend="gloo", timeout=datetime.timedelta(seconds=timeout)
    )
    train_dataloader.dataset.set_epoch(epoch)

    for batch_idx, batch in tqdm(
        enumerate(train_dataloader),
        desc="Train",
        disable=dist.get_rank() != 0,
    ):
        model.train()

        if batch_idx > 0:
            try:
                dist.monitored_barrier(
                    group=dist_group, timeout=datetime.timedelta(seconds=timeout)
                )
            except:
                break

        loss = forward(model, ref_model, batch, dpo_loss, dtype, device)
        backward(loss, model)
        update_params(model)

        if training_config["step"] < scheduler.last_epoch:
            training_config["step"] = scheduler.last_epoch
            if eval_steps > 0 and (training_config["step"] + 1) % eval_steps == 0:
                loss_dict = eval_func(model, dev_dataloader)
                logger.info(f"Training -- loss_dict: {loss_dict}")
                if saved_dir is not None:
                    save_model(model, training_config, saved_dir, "training", loss_dict)

    dist.destroy_process_group(dist_group)
