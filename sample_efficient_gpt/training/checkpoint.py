from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer

from sample_efficient_gpt.config_schema import Config
from sample_efficient_gpt.utils.config_tools import save_config


def save_checkpoint(
    fpath: Path,
    cfg: Config | None,
    model: nn.Module,
    optimizers: list[Optimizer],
    iteration: int = 0,
    run_id: str | None = None,
):
    torch.save(
        {
            "config": save_config(cfg) if cfg is not None else None,
            "model": model.state_dict(),
            "optimizer": [optimizer.state_dict() for optimizer in optimizers],
            "iteration": iteration,
            "run_id": run_id,
        },
        fpath,
    )


def load_checkpoint(fpath: Path, model: nn.Module, optimizers: list[Optimizer] | None, device: str) -> int:
    checkpoint = torch.load(fpath, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizers is not None:
        # for muon we load 2 optimizers
        for opt, opt_sd in zip(optimizers, checkpoint["optimizer"]):
            opt.load_state_dict(opt_sd)
    return checkpoint["iteration"]
