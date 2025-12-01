"""Compute per-sample loss for SFT data. Supports torchrun sharding."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Compute token loss over chat data.")
    parser.add_argument("--checkpoint", required=False, help="HF causal LM checkpoint.")
    parser.add_argument("--data-file", required=False, help="JSONL dataset with fields input/messages and output.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for loss shards.")
    parser.add_argument("--output-prefix", type=str, default="loss", help="Prefix for rank-suffixed outputs.")
    parser.add_argument("--max-length", type=int, default=1024, help="Truncate sequences to this length.")
    parser.add_argument("--world-size", type=int, default=None, help="Override world size (else torchrun env).")
    parser.add_argument("--rank", type=int, default=None, help="Override rank (else LOCAL_RANK/WORLD_SIZE).")
    parser.add_argument("--start", type=int, default=None, help="Optional dataset start index before sharding.")
    parser.add_argument("--end", type=int, default=None, help="Optional dataset end index before sharding.")
    parser.add_argument("--task", type=str, default="infer", help="infer | filter -- task")
    parser.add_argument("--input-path", type=str, default=None, help="dataset jsonl for filtering by loss")
    parser.add_argument("--output-path", type=str, default=None, help="save path after filtering")
    parser.add_argument("--keep-pct", type=float, default=0.3, help="% for filtering")
    return parser.parse_args()


def get_model(checkpoint, device):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        # TODO: make sure safetensors is enabled
        # use_safetensors=False,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer


def tokenize(tokenizer, conv):
    tokenized_output = tokenizer.apply_chat_template(
        conv,
        return_assistant_tokens_mask=True,
        return_dict=True,
        return_tensors="np",
        add_generation_prompt=False,
    )
    labels = tokenized_output.pop("assistant_masks")
    labels[labels == 0] = -100
    return {
        "input_ids": tokenized_output["input_ids"][0],
        "labels": labels[0],
        "attention_mask": tokenized_output["attention_mask"][0],
    }


def get_sample_loss(input_ids, logits, labels, loss_fn) -> float:
    inputs = logits[0, :-1]
    targets = input_ids[0, 1:]
    with torch.no_grad():
        loss = loss_fn(inputs, targets)[labels[: len(input_ids[0]) - 1] != -100].detach().mean().item()
        return loss


def filter_dataset_by_loss(input_path, output_path, keep_pct=0.25):
    ds = load_dataset("json", data_files=input_path)['train']
    losses = np.array(ds['loss']).astype(float)
    indices = np.arange(len(losses))
    mask = ~np.isnan(losses)
    losses = losses[mask]
    indices = indices[mask]

    loss_thresh = np.percentile(losses, int(keep_pct * 100))
    print(f"using loss threshold {loss_thresh=}")

    new_mask1 = losses < 0.95 * loss_thresh
    new_mask2 = (losses < 1.75 * loss_thresh) & (losses > 1.5 * loss_thresh)
    indices = np.concat([indices[new_mask1], indices[new_mask2]])
    new_ds = ds.select(indices)
    print(f"Keeping {len(new_ds)/len(ds)} pct of samples")
    new_ds.to_json(output_path)


def infer(args):
    rank = args.rank if args.rank is not None else int(os.environ.get("LOCAL_RANK", 0))
    world_size = args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    model, tokenizer = get_model(args.checkpoint, device)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    ds = load_dataset("json", data_files=args.data_file)["train"]
    total = len(ds)
    subset_start = args.start or 0
    subset_end = args.end if args.end is not None else total
    ds = ds.select(range(subset_start, min(subset_end, total)))

    total = len(ds)
    per_rank = (total + world_size - 1) // world_size
    shard_start = rank * per_rank
    shard_end = min(total, shard_start + per_rank)

    if shard_start >= total:
        raise ValueError(f"Rank {rank} has no data to process (start {shard_start} >= total {total}).")

    ds = ds.select(range(shard_start, shard_end))

    results = []
    for row in tqdm(ds):
        conv_no_assistant = row["input"]["messages"]
        target = row["output"]
        conv = conv_no_assistant + [{"role": "assistant", "content": target}]
        tokenized = tokenize(tokenizer, conv)

        input_ids, attention_mask, labels = tokenized["input_ids"], tokenized["attention_mask"], tokenized["labels"]
        input_ids = input_ids[: args.max_length]
        attention_mask = attention_mask[: args.max_length]

        input_ids = torch.tensor(input_ids, device=device, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask, device=device, dtype=torch.long).unsqueeze(0)
        with torch.autocast("cuda", torch.bfloat16), torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
            loss = get_sample_loss(input_ids, logits, labels, loss_fn)
        results.append(
            {
                "index": row.get("index"),
                "input": {"messages": conv_no_assistant},
                "output": target,
                "loss": loss,
            }
        )
    new_ds = Dataset.from_list(results)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_path = args.output_dir / f"{args.output_prefix}_rank{rank}.jsonl"
    new_ds.to_json(output_path)


def main():
    args = parse_args()
    if args.task == "infer":
        infer()
    else:
        filter_dataset_by_loss(args.input_path, args.output_path, args.keep_pct)
    


if __name__ == "__main__":
    main()
