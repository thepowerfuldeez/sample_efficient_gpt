"""
Script to filter txt dataset using all available gpus

run with uv run torchrun --nproc_per_node 4 edu_filter_txt.py 
"""

import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from pathlib import Path
from typing import BinaryIO
from sample_efficient_gpt.tokenizer.pretokenization import find_chunk_boundaries


def apply_classifier_language_batch(text, batch_size=128):
    docs = text.split("<|endoftext|>")[:-1]
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
        if res["int_score"] >= 2:
            row = {
                "text": doc,
                "length": len(doc),
                **res,
            }
            rows.append(row)

    print(f"after filtering: {len(rows)} results")
    return rows


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_default_device(f"cuda:{rank}")

    filepath = "/home/george/datasets/finepdfs_en_2020plus_100b/all.txt"
    output_dir = "/mnt/harddrive/datasets/finepdfs_en_2020plus_edu/data"
    special_token = "<|endoftext|>"
    num_chunks = 512

    handle: BinaryIO = Path(filepath).open("rb")
    boundaries = find_chunk_boundaries(handle, num_chunks, special_token.encode())
    handle.close()

    # process only first 20B
    boundaries = boundaries[:105]
    bounds = list(zip(boundaries[:-1], boundaries[1:]))
    items = [(i, bound) for i, bound in enumerate(bounds)]
    import random
    random.shuffle(items)

    local_chunk_size = len(items) // world
    rank_items = items[rank * local_chunk_size : rank * local_chunk_size + local_chunk_size]

    repo_name = "HuggingFaceFW/fineweb-edu-classifier"
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    model = AutoModelForSequenceClassification.from_pretrained(repo_name)

    f = Path(filepath).open("rb")
    for chunk_i, (start, end) in tqdm(rank_items, desc="Boundaries"):
        out_path = f"{output_dir}/train-{chunk_i:06d}.parquet"
        if Path(out_path).exists():
            print(f"{out_path} exists; continue")
            continue
        f.seek(start)
        chunk = f.read(end-start).decode("utf-8", errors="ignore")
        results = apply_classifier_language_batch(chunk, batch_size=128)
        ds = Dataset.from_list(results)
        ds.to_parquet(out_path)
        print(f"Successfully wrote to {out_path}")
    f.close()
    dist.destroy_process_group()
