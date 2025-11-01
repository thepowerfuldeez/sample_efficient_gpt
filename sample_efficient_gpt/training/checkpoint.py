import random
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer

from torch.distributed.fsdp import FSDPModule
from torch.distributed.checkpoint.state_dict import set_model_state_dict, set_optimizer_state_dict, StateDictOptions
from sample_efficient_gpt.config_schema import Config
from sample_efficient_gpt.utils.config_tools import save_config


def sharded_sd_to_full(sharded_sd):
    cpu_state_dict = {}
    for param_name, sharded_param in sharded_sd.items():
        full_param = sharded_param.full_tensor()
        # target log rank
        if torch.distributed.get_rank() == 2:
            cpu_state_dict[param_name] = full_param.cpu()
        else:
            del full_param
    return cpu_state_dict


def save_checkpoint(
    fpath: Path,
    cfg: Config | None,
    model: nn.Module | FSDPModule,
    optimizers: list[Optimizer],
    iteration: int = 0,
    run_id: str | None = None,
):
    if cfg.trainer.dist_mode == "fsdp" and dist.is_initialized():
        model_state_dict = sharded_sd_to_full(model.state_dict())
        optimizers_state_dict = [sharded_sd_to_full(opt.state_dict()) for opt in optimizers]
    else:
        model_state_dict = model.state_dict()
        optimizers_state_dict = [optimizer.state_dict() for optimizer in optimizers]
    torch_state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }

    torch.save(
        {
            "config": save_config(cfg) if cfg is not None else None,
            "model": model_state_dict,
            "optimizer": optimizers_state_dict,
            "iteration": iteration,
            "run_id": run_id,
            "rng_state": torch_state,
        },
        fpath,
    )


def load_checkpoint(fpath: Path, cfg: Config, model: nn.Module, optimizers: list[Optimizer] | None) -> int:
    checkpoint = torch.load(fpath, map_location="cpu", weights_only=False)
    if cfg.trainer.dist_mode == "fsdp" and dist.is_initialized():
        set_model_state_dict(
            model,
            checkpoint["model"],
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )
        if optimizers is not None:
            for opt, opt_sd in zip(optimizers, checkpoint["optimizer"]):
                set_optimizer_state_dict(
                    model,
                    opt,
                    opt_sd,
                    options=StateDictOptions(
                        full_state_dict=True,
                        broadcast_from_rank0=True,
                    ),
                )
    else:
        model.load_state_dict(checkpoint["model"])
        if optimizers is not None:
            # for muon we load 2 optimizers
            for opt, opt_sd in zip(optimizers, checkpoint["optimizer"]):
                for ((k1, v1), (k2, v2)) in zip(opt.state_dict()['state'], opt_sd['state']):
                    assert k1 == k2 and v1.shape == v2.shape, f"{k1} != {k2} or {v1.shape=}!={v2.shape=}"
                opt.load_state_dict(opt_sd)

    rng = checkpoint.get("rng_state", None)
    if rng:
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    return checkpoint["iteration"], checkpoint.get("run_id")
