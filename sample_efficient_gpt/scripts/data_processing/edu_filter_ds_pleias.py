"""
Script to filter parquet dataset using all available gpus

run with uv run torchrun --nproc_per_node 4 edu_filter_ds_pleias.py
"""

import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, disable_caching, concatenate_datasets
from pathlib import Path
from typing import BinaryIO
from sample_efficient_gpt.tokenizer.pretokenization import find_chunk_boundaries

disable_caching()


def apply_classifier_language_batch(docs, batch_size=128):
    print(f"received {len(docs)} docs")
    results = []
    with torch.autocast("cuda", torch.bfloat16), torch.inference_mode():
        for i in tqdm(list(range(0, len(docs), batch_size)), desc="Batches"):
            texts_batch = docs[i : i + batch_size]
            inputs = tokenizer(texts_batch, return_tensors="pt", padding="longest", truncation=True)
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).float().detach().cpu().numpy()
            scores = [logit.item() for logit in logits]
            for score in scores:
                result = {
                    "score": score,
                    "int_score": int(round(max(0, min(score, 5)))),
                }
                results.append(result)
    rows = []
    for res, doc in zip(results, docs):
        if res["int_score"] >= 3:
            row = {
                "text": doc,
                "length": len(doc),
                **res,
            }
            rows.append(row)

    print(f"after filtering: {len(rows)} results")
    return rows


def read_parquet(chunk_path):
    ds = load_dataset("parquet", data_files=str(chunk_path), split="train")
    ds_memo = ds.filter(lambda x: x["exercise"] == "memorization" and x["language"] == "en", num_proc=8)
    ds_memo = ds_memo.select(range(0, int(0.25 * len(ds_memo))))
    ds_other = ds.filter(
        lambda x: x["exercise"] in {"mcq", "math mcq", "arithmetics", "creative writing"} and x["language"] == "en",
        num_proc=8,
    )
    ds = concatenate_datasets([ds_memo, ds_other])
    ds = ds.map(lambda x: {"text": x["query"] + "\n<think>" + x["synthetic_reasoning"] + "</think>\n" + x["synthetic_answer"]})
    return ds.shuffle(seed=42)


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_default_device(f"cuda:{rank}")

    filepath = "/mnt/harddrive/datasets/pleias_synth/data/"
    output_dir = "/mnt/harddrive/datasets/pleias_synth/data_filtered"
    Path(output_dir).mkdir(exist_ok=True)

    parquet_files = list(Path(filepath).glob("*.parquet"))

    local_chunk_size = len(parquet_files) // world
    local_parquet_files = parquet_files[rank * local_chunk_size : rank * local_chunk_size + local_chunk_size]
    print(f"[{rank=}] processing {len(local_parquet_files)} files")

    repo_name = "HuggingFaceFW/fineweb-edu-classifier"
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    model = AutoModelForSequenceClassification.from_pretrained(repo_name)

    for fp in local_parquet_files:
        out_path = f"{output_dir}/{fp.name}"
        if Path(out_path).exists():
            print("out_path exists", out_path)
            continue
        docs = read_parquet(fp)["text"]
        results = apply_classifier_language_batch(docs, batch_size=128)
        ds = Dataset.from_list(results)
        ds.to_parquet(out_path)
        print(f"Successfully wrote to {out_path}")
    dist.destroy_process_group()
