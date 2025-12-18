from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer

from sample_efficient_gpt.config_schema import Config
from sample_efficient_gpt.utils.config_tools import save_config


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        t = [_to_device(xx, device) for xx in x]
        return type(x)(t)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x  # ints/floats/etc.


def save_checkpoint(
    fpath: Path,
    cfg: Config | None,
    model: nn.Module,
    optimizers: list[Optimizer],
    iteration: int = 0,
    run_id: str | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    shard_type: str | None = None,
):
    model_state_dict = model.state_dict()
    optimizers_state_dict = [optimizer.state_dict() for optimizer in optimizers]

    torch.save(
        {
            "config": save_config(cfg) if cfg is not None else None,
            "model": model_state_dict,
            "optimizer": optimizers_state_dict,
            "iteration": iteration,
            "run_id": run_id,
            "rank": rank,
            "world_size": world_size,
            "shard_type": shard_type,
        },
        fpath,
    )


def load_model(fpath: Path, cfg: Config, model: nn.Module) -> int:
    checkpoint = torch.load(fpath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return checkpoint["iteration"], checkpoint.get("run_id")


def load_model_expert_parallel(
    fpath: Path, cfg: Config, model: nn.Module, *, rank: int, world_size: int
) -> tuple[int, str | None]:
    """
    Load a checkpoint into an expert-parallel MoE model.

    Supports:
    - Loading a per-rank sharded checkpoint saved with shard_type='expert_parallel'
    - Loading a full (non-sharded) checkpoint and slicing experts by rank:
        global_expert = rank * num_local_experts + local_expert
    """
    checkpoint = torch.load(fpath, map_location="cpu", weights_only=False)
    src_sd: dict[str, torch.Tensor] = checkpoint["model"]
    shard_type = checkpoint.get("shard_type")

    if shard_type == "expert_parallel":
        model.load_state_dict(src_sd)
        return checkpoint["iteration"], checkpoint.get("run_id")

    num_experts = int(cfg.model.moe_num_experts)
    if num_experts <= 0:
        raise ValueError("load_model_expert_parallel requires cfg.model.moe_num_experts > 0")
    if num_experts % world_size != 0:
        raise ValueError(f"moe_num_experts ({num_experts}) must be divisible by world_size ({world_size})")
    num_local = num_experts // world_size

    dst_sd = model.state_dict()
    remapped: dict[str, torch.Tensor] = {}
    for k, v in dst_sd.items():
        parts = k.split(".")
        # blocks.{layer}.ffn.experts.{local}.*
        if len(parts) >= 6 and parts[0] == "blocks" and parts[2] == "ffn" and parts[3] == "experts" and parts[4].isdigit():
            local = int(parts[4])
            if not (0 <= local < num_local):
                raise ValueError(f"Local expert index out of range for rank shard: {k}")
            global_expert = rank * num_local + local
            src_k = ".".join(parts[:4] + [str(global_expert)] + parts[5:])
            if src_k not in src_sd:
                raise KeyError(f"Checkpoint missing expected global expert key: {src_k}")
            remapped[k] = src_sd[src_k]
        else:
            if k not in src_sd:
                raise KeyError(f"Checkpoint missing expected key: {k}")
            remapped[k] = src_sd[k]

    model.load_state_dict(remapped)
    return checkpoint["iteration"], checkpoint.get("run_id")


def load_optimizer(fpath: Path, cfg: Config, model, optimizers: list[Optimizer] | None) -> None:
    checkpoint = torch.load(fpath, map_location="cpu", weights_only=False)
    if optimizers is not None and optimizers[0] is not None:
        if checkpoint.get("optimizer") is None or (
            isinstance(checkpoint["optimizer"], list) and checkpoint["optimizer"][0] is None
        ):
            print("no optimizers detected in the checkpoint, skipping...")
            return
        # for muon we load 2 optimizers
        for opt, opt_sd in zip(optimizers, checkpoint["optimizer"]):
            for (k1, v1), (k2, v2) in zip(opt.state_dict()["state"], opt_sd["state"]):
                assert k1 == k2 and v1.shape == v2.shape, f"{k1} != {k2} or {v1.shape=}!={v2.shape=}"
            opt.load_state_dict(opt_sd)
            for p, state in opt.state.items():  # keys are parameter tensors
                p_dev = p.device
                for k, v in list(state.items()):
                    state[k] = _to_device(v, p_dev)
