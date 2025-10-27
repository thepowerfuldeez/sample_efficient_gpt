import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

from sample_efficient_gpt.configs.gpt_small_faster import cfg
from sample_efficient_gpt.config_schema import Config
from sample_efficient_gpt.utils.config_tools import dataclass_to_nested_dict, wandb_run_name
from sample_efficient_gpt.training.trainer import Trainer
from sample_efficient_gpt.utils.logger import logger
from sample_efficient_gpt.utils.config_tools import apply_overrides

BACKEND = "nccl"


def shutdown():
    dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", type=str, required=True)
    p.add_argument("--validation-path", type=str, required=True)
    p.add_argument("--load-from", type=str, help='resume ckpt', default=None)
    p.add_argument("--override", type=str, help='{"k": "v"} override to the cfg as dict')
    p.add_argument("--config", type=str, default="gpt_small_faster", help="config path, default: gpt_small_faster")
    p.add_argument("--world-size", type=int, default=1)
    return p.parse_args()


def train(rank, cfg: Config, args):
    if args.override:
        override = json.loads(args.override)
        if "optim.seesaw_steps" in override:
            override['optim.seesaw_steps'] = eval(override['optim.seesaw_steps'])
    else:
        override = {}
    s = ""
    for k, v in override.items():
        s += f"{k.split('.')[-1][:4]}={str(v)[:4]}"
    if "run_name" in args.override:
        # no need to extract k-v pairs if specified run name manually
        cfg.trainer.run_name = override["trainer.run_name"]
        run_name: str = wandb_run_name(cfg)
        print("run name overriden")
    else:
        run_name: str = wandb_run_name(cfg)[:47]
        run_name = f"{run_name}_{s}"
    logger.info(f"Training run: {run_name}")

    override = {
        **override,
        "data.train_path": args.train_path,
        "data.validation_path": args.validation_path,
    }

    cfg = apply_overrides(cfg, override)
    overrides = {"optim.cosine_steps": cfg.trainer.max_steps}
    cfg = apply_overrides(cfg, overrides)
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
    if args.world_size > 1:
        dist.init_process_group(BACKEND)
        if BACKEND != "gloo":
            rank = dist.get_rank()
            torch.cuda.set_device(f"cuda:{rank}")
        train(rank, cfg, args)
    else:
        train(0, cfg, args)
