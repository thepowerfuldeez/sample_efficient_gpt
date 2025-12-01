"""Compute embeddings for prompt+response pairs for FLA filtering."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings for chat pairs.")
    parser.add_argument("--data-file", required=True, help="JSONL dataset with input/messages and output.")
    parser.add_argument(
        "--model-checkpoint",
        default="Snowflake/snowflake-arctic-embed-l-v2.0",
        help="Embedding model checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size for embedding model.")
    parser.add_argument("--world-size", type=int, default=None, help="Override world size (else torchrun env).")
    parser.add_argument("--rank", type=int, default=None, help="Override rank (else LOCAL_RANK/WORLD_SIZE).")
    parser.add_argument("--start", type=int, default=None, help="Optional dataset start index before sharding.")
    parser.add_argument("--end", type=int, default=None, help="Optional dataset end index before sharding.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for embedding shards.")
    parser.add_argument("--output-prefix", type=str, default="embedding", help="Prefix for rank-suffixed outputs.")
    return parser.parse_args()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_model(device, checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map=device,
    )
    return model, tokenizer


def main():
    args = parse_args()

    rank = args.rank if args.rank is not None else int(os.environ.get("LOCAL_RANK", 0))
    world_size = args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", 1))

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    model, tokenizer = get_model(device, args.model_checkpoint)

    ds = load_dataset("json", data_files=args.data_file)["train"]
    total = len(ds)
    subset_start = args.start or 0
    subset_end = args.end if args.end is not None else total
    ds = ds.select(range(subset_start, min(subset_end, total)))

    total = len(ds)
    per_rank = (total + world_size - 1) // world_size
    shard_start = rank * per_rank
    shard_end = min(total, shard_start + per_rank)
    ds = ds.select(range(shard_start, shard_end))

    results = []
    for i in tqdm(range(0, len(ds), args.batch_size)):
        batch_rows = ds.select(range(i, min(len(ds), i + args.batch_size)))
        texts = [
            "\n".join([msg["content"] for msg in row["input"]["messages"]] + [row["output"]]) for row in batch_rows
        ]

        with torch.autocast("cuda", torch.bfloat16) if torch.cuda.is_available() else torch.autocast(
            "cpu", torch.bfloat16
        ):
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                model_output = model(**inputs)
            sentence_embeddings = mean_pooling(model_output, inputs["attention_mask"])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        results.extend(sentence_embeddings.cpu().numpy().tolist())

    ds = ds.add_column("embedding", results)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_path = args.output_dir / f"{args.output_prefix}_rank{rank}.jsonl"
    ds.to_json(output_path)


if __name__ == "__main__":
    main()
