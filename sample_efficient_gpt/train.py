import argparse
import importlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
import wandb

from sample_efficient_gpt.config_schema import Config
from sample_efficient_gpt.training.trainer import Trainer
from sample_efficient_gpt.utils.config_tools import (
    apply_overrides,
    dataclass_to_nested_dict,
    load_config_from_yaml,
    wandb_run_name,
)
from sample_efficient_gpt.utils.logger import logger

BACKEND = "nccl"


def shutdown():
    dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", type=str)
    p.add_argument("--validation-path", type=str)
    p.add_argument("--load-from", type=str, help="resume ckpt", default=None)
    p.add_argument("--override", type=str, help='{"k": "v"} override to the cfg as dict')
    p.add_argument(
        "--config",
        type=str,
        default="gpt_small_faster",
        help="Config module (sample_efficient_gpt.configs.<name>) or YAML path.",
    )
    p.add_argument(
        "--config-key",
        type=str,
        default=None,
        help="Optional key when loading a YAML file with multiple experiments.",
    )
    p.add_argument("--world-size", type=int, default=1)
    return p.parse_args()


def load_cfg(config_path: str, config_key: str | None) -> Config:
    if config_path.endswith((".yaml", ".yml")):
        return load_config_from_yaml(config_path, config_key)
    module_name = config_path
    if not module_name.startswith("sample_efficient_gpt.configs"):
        module_name = f"sample_efficient_gpt.configs.{module_name}"
    module = importlib.import_module(module_name)
    cfg = getattr(module, "cfg", None)
    if cfg is None:
        raise ValueError(f"Config module {module_name} does not define cfg")
    return cfg


def train(rank, cfg: Config, args):
    if args.override:
        override = json.loads(args.override)
        if "optim.seesaw_steps" in override:
            override["optim.seesaw_steps"] = eval(override["optim.seesaw_steps"])
    else:
        override = {}
    cli_overrides = {
        "data.train_path": args.train_path,
        "data.validation_path": args.validation_path,
        "trainer.load_from": args.load_from,
    }
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
    override = {**override, **cli_overrides}

    summary_suffix = "_".join(f"{k.split('.')[-1][:4]}={str(v)[:4]}" for k, v in override.items())
    override_run_name = override.get("trainer.run_name")
    if override_run_name:
        cfg.trainer.run_name = override_run_name
        run_name: str = wandb_run_name(cfg)
        print("run name overriden")
    else:
        run_name: str = wandb_run_name(cfg)[:47]
        if summary_suffix:
            run_name = f"{run_name}_{summary_suffix}"
    logger.info(f"Training run: {run_name}")

    cfg = apply_overrides(cfg, override)
    if not cfg.data.train_path or not cfg.data.validation_path:
        raise ValueError("Both train and validation paths must be provided via config or CLI overrides.")
    overrides = {"optim.cosine_steps": cfg.trainer.max_steps}
    cfg = apply_overrides(cfg, overrides)
    cfg.trainer.save_dir = Path(cfg.trainer.save_dir)
    cfg.trainer.save_dir = cfg.trainer.save_dir / run_name

    if (args.world_size > 1 and rank == 2) or (args.world_size == 1 and rank == 0):
        run = wandb.init(project=cfg.project, name=run_name, config=dataclass_to_nested_dict(cfg))
    else:
        run = None
    trainer = Trainer(cfg, load_from=args.load_from, wandb=run)
    trainer.train()

    if (args.world_size > 1 and rank == 2) or (args.world_size == 1 and rank == 0):
        run.finish()
    if args.world_size > 1:
        shutdown()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_cfg(args.config, args.config_key)
    if args.world_size > 1:
        dist.init_process_group(BACKEND)
        if BACKEND != "gloo":
            rank = dist.get_rank()
            torch.cuda.set_device(f"cuda:{rank}")
        train(rank, cfg, args)
    else:
        train(0, cfg, args)
