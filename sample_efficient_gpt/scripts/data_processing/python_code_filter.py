"""
Script to filter txt dataset using all available gpus

run with uv run torchrun --nproc_per_node 4 python_code_filter.py
"""

import os
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from pathlib import Path
from typing import BinaryIO
from sample_efficient_gpt.tokenizer.pretokenization import find_chunk_boundaries


def apply_classifier_language_batch(row, batch_size=2):
    files = row["text"].split("<|endoftext|>")[:-1]
    results_per_file = []
    with torch.autocast("cuda", torch.bfloat16), torch.inference_mode():
        for i in range(0, len(files), batch_size):
            texts_batch = files[i : i + batch_size]
            inputs = tokenizer(texts_batch, return_tensors="pt", padding="longest", truncation=True)
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).float().detach().cpu().numpy()
            scores = [logit.item() for logit in logits]
            for score in scores:
                result = {
                    "score": score,
                    "int_score": int(round(max(0, min(score, 5)))),
                }
                results_per_file.append(result)
    new_text = ""
    out_results = []
    for result, file in zip(results_per_file, files):
        if result["int_score"] > 2:
            new_text += file
            new_text += "<|endoftext|>"
            out_results.append(result)

    return {
        "text": new_text,
        "results": out_results,
        "length": len(new_text),
        "repo_name": row["repo_name"],
    }


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.getenv("LOCAL_RANK"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")

    output_dir = "/home/george/datasets/the_stack_v2_updated/data_filtered/"

    if rank == 0:
        ds_stack_meta = load_dataset("thepowerfuldeez/the-stack-v2-train-smol-ids-updated")["train"]
        ds_stack_meta_python = ds_stack_meta.filter(lambda x: x["gha_language"] == "Python", num_proc=8)
        ds_stack_meta_python2 = ds_stack_meta.filter(
            lambda x: sum(a["language"] == "Python" for a in x["files"]) / len(x["files"]) > 0.75
            and x["gha_language"] != "Python",
            num_proc=8,
        )
        python_repos = list(ds_stack_meta_python["repo_name"]) + list(ds_stack_meta_python2["repo_name"])
        python_repos = set(python_repos)
        ds_python = load_dataset("/home/george/datasets/the_stack_v2_updated/data")["train"]
        ds_python = ds_python.filter(lambda x: x["repo_name"] in python_repos)
        # ds_python = ds_python.map(lambda x: {"original_length": len(x["text"])}, num_proc=8)
        shards = [ds_python.shard(world, i) for i in range(world)]
    out = [None]  # one slot per receiving rank; scatter will fill rank's slot
    dist.scatter_object_list(out, shards if rank == 0 else None, src=0)

    # local_chunk_size = len(ds_python) // world
    # rank_slice = range(rank * local_chunk_size, rank * local_chunk_size + local_chunk_size)
    # ds_local = ds_python.select(rank_slice)
    ds_local = out[0]

    repo_name = "HuggingFaceTB/stack-edu-classifier-python"
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            repo_name,
            device_map=None,
        )
        .to(device, dtype=torch.bfloat16)
        .eval()
    )

    # 1 chunk is already prepared
    offset = 1
    ds_rows = []
    for row in tqdm(
        ds_local,
        disable=not dist.is_initialized() or dist.get_rank() != 0,
    ):
        try:
            row = apply_classifier_language_batch(row, 128)
            ds_rows.append(row)
        except:
            print("error occured")
            continue
    chunk_i = offset + rank * len(ds_local)
    out_path = f"{output_dir}/train-{chunk_i:06d}.parquet"
    Dataset.from_list(ds_rows).to_parquet(out_path)
    print(f"Successfully wrote to {out_path}")
    dist.destroy_process_group()