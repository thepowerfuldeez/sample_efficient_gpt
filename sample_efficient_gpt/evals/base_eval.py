"""
Evlauate the CORE metric for a given model.

Run on a single GPU:
python base_eval.py

Run with torchrun on e.g. 8 GPUs:
torchrun --nproc_per_node=8 base_eval.py

The script will print the CORE metric to the console.
"""

import os
import sys
import time
import json
import random
import yaml
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import torch
import wandb

import torch.distributed as dist
from sample_efficient_gpt.evals.core_eval import evaluate_task
from sample_efficient_gpt.training.trainer import Trainer


def print0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def evaluate_model(base_dir: Path, model, tokenizer, device, max_per_task=-1):
    """
    Evaluate a base model on the CORE benchmark.
    - max_per_task: crop the data to this many examples per task for testing (-1 = disable)
    TODO: clean up this function, delete the need for all the files, for pandas dependency, etc.
    """
    # Load config and task metadata
    eval_bundle_dir = base_dir / "eval_bundle"
    config_path = eval_bundle_dir / "core.yaml"
    data_base_path = eval_bundle_dir / "eval_data"
    eval_meta_data = eval_bundle_dir / "eval_meta_data.csv"
    config = yaml.safe_load(config_path.read_text())
    tasks = config["icl_tasks"]
    eval_metadata = pd.read_csv(eval_meta_data)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.monotonic()
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end="")

        # Load data for this task
        data_path = data_base_path / task_meta["dataset_uri"]
        with data_path.open() as f:
            data = [json.loads(line.strip()) for line in f]

        # shuffle the data because in many cases it appears ordered but we want
        # the abillity to only run a subset of the data for debugging purposes etc.
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # run the evaluation for this task
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        row = eval_metadata[eval_metadata["Eval Task"] == label]
        random_baseline = row["Random baseline"].values[0]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.monotonic()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {"results": results, "centered_results": centered_results, "core_metric": core_metric}
    return out


# -----------------------------------------------------------------------------
# HuggingFace loading utilities and light wrappers for a model

# we need logits from the model


def parse_args():
    p = ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--base_dir", default=Path("/home/george/.cache/sample_efficient_gpt"), type=Path)
    return p.parse_args()


# -----------------------------------------------------------------------------
def main():
    # run with torchrun
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    device = f"cuda:{rank}"
    args = parse_args()

    trainer = Trainer(load_from=args.checkpoint, load_components="infer", **{"trainer.device": device})
    run_id = trainer.run_id
    project = trainer.cfg.project
    model = trainer.model
    tokenizer = trainer.tokenizer
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    base_dir = args.base_dir

    model_name = f"base_model (step {trainer.iteration})"  # just for logging
    model_slug = f"base_model_{trainer.iteration:06d}"  # for the output csv file

    # Evaluate the model
    with autocast_ctx:
        out = evaluate_model(base_dir, model, tokenizer, device)

    # Write out the results to a csv file
    core_metric = None
    centered_results = {}
    if rank == 0:
        output_csv_path = base_dir / "base_eval" / f"{model_slug}.csv"
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results = out["results"]
        centered_results = out["centered_results"]
        core_metric = out["core_metric"]
        with open(output_csv_path, "w") as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
        # Print the content of the csv file to console too
        print0("=" * 80)
        print0(f"Model: {model_name}")
        print0("=" * 80)
        with open(output_csv_path) as f:
            print0(f.read())

        with wandb.init(project=project, resume="must", id=run_id) as run:
            run.log({"eval/core_metric": core_metric})

    # Log to report
    # from nanochat.report import get_report

    # get_report().log(
    #     section="Base model evaluation",
    #     data=[
    #         {
    #             "Model": model_name,
    #             "CORE metric": core_metric,
    #         },
    #         centered_results,  # the full table
    #     ],
    # )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
