import argparse
import json
from pathlib import Path

import wandb

from sample_efficient_gpt.configs.gpt_small_faster import cfg
from sample_efficient_gpt.config_schema import Config
from sample_efficient_gpt.utils.config_tools import dataclass_to_nested_dict, wandb_run_name
from sample_efficient_gpt.training.trainer import Trainer
from sample_efficient_gpt.utils.logger import logger
from sample_efficient_gpt.utils.config_tools import apply_overrides


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-path",
        type=str,
        default=str(Path(__file__).parent.parent / "data_tokenized/TinyStoriesV2-GPT4-train.npy"),
    )
    p.add_argument(
        "--validation-path",
        type=str,
        default=str(Path(__file__).parent.parent / "data_tokenized/TinyStoriesV2-GPT4-valid.npy"),
    )
    p.add_argument("--override", type=str, help='{"k": "v"} override to the cfg as dict')
    p.add_argument("--config", type=str, default="gpt_small_faster", help="config path, default: gpt_small_faster")
    return p.parse_args()


def train(cfg: Config, args):
    if args.override:
        override = json.loads(args.override)
    else:
        override = {}
    s = ""
    for k, v in override.items():
        s += f"{k.split('.')[-1][:4]}={str(v)[:4]}"
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
    cfg.trainer.save_dir = cfg.trainer.save_dir / s

    run = wandb.init(project=cfg.project, name=run_name, config=dataclass_to_nested_dict(cfg))
    trainer = Trainer(cfg, wandb=run)
    trainer.train()

    run.finish()


def test(cfg: Config):
    import torch

    trainer = Trainer(cfg)
    print(trainer.generate(torch.tensor([0, 1, 2]), 5, top_p=0.8, temperature=0.1))


train(cfg, parse_args())
# test(cfg)
