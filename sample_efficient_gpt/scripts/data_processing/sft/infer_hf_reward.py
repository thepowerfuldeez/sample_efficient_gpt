"""Reward classifier. Shard across ranks when launched with torchrun."""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_model(device, checkpoint):
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model.eval()
    return model, tokenizer


def format_conversations(tokenizer, conversations):
    convs_fmt = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in conversations]
    convs_fmt_fixed = []
    for conv_fmt in convs_fmt:
        if tokenizer.bos_token is not None and conv_fmt.startswith(tokenizer.bos_token):
            conv_fmt = conv_fmt[len(tokenizer.bos_token) :]
        convs_fmt_fixed.append(conv_fmt)
    return convs_fmt_fixed


def main():
    parser = argparse.ArgumentParser(description="Run reward model inference with torchrun.")
    parser.add_argument("--data-file", required=True, help="JSONL dataset produced by infer_hf_k.py")
    parser.add_argument(
        "--model-checkpoint",
        default="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        help="Reward model checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Rows to score per forward pass.")
    parser.add_argument("--max-length", type=int, default=1024, help="Max tokens per example.")
    parser.add_argument("--world-size", type=int, default=None, help="Override world size (else torchrun env).")
    parser.add_argument("--rank", type=int, default=None, help="Override rank (else LOCAL_RANK/WORLD_SIZE).")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="k_rewards",
        help="Prefix for rank-suffixed JSONL outputs.",
    )
    parser.add_argument(
        "--start_idx", type=int, default=-1, help="Number of rows to generate (uses ds.select(range(limit)))."
    )
    parser.add_argument(
        "--end_idx", type=int, default=-1, help="Number of rows to generate (uses ds.select(range(limit)))."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for outputs.")
    args = parser.parse_args()

    rank = args.rank if args.rank is not None else int(os.environ.get("LOCAL_RANK", 0))
    world_size = args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", 1))

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    model, tokenizer = get_model(device, args.model_checkpoint)
    ds = load_dataset("json", data_files=args.data_file)["train"]

    if args.start_idx != -1 and args.end_idx != -1:
        if args.end_idx == -1:
            end = len(ds)
        else:
            end = args.end_idx
        ds = ds.select(range(args.start_idx, end))

    total = len(ds)
    per_rank = (total + world_size - 1) // world_size
    start = rank * per_rank
    end = min(total, start + per_rank)

    if start >= total:
        raise ValueError(f"Rank {rank} has no data to process (start {start} >= total {total}).")

    ds_sample = ds.select(range(start, end))

    results = []
    for batch_start in tqdm(range(0, len(ds_sample), args.batch_size)):
        batch_end = min(batch_start + args.batch_size, len(ds_sample))
        batch = ds_sample.select(range(batch_start, batch_end))
        conversations = []
        conv_lengths = []
        for row in batch:
            inp = row["input"]["messages"]
            candidates = [row["output"]] + row["generated_outputs"]
            convs = [inp + [{"role": "assistant", "content": cand}] for cand in candidates]
            conversations.extend(convs)
            conv_lengths.append(len(convs))

        with (
            torch.autocast("cuda", torch.bfloat16)
            if torch.cuda.is_available()
            else torch.autocast("cpu", torch.bfloat16),
            torch.no_grad(),
        ):
            convs_fmt_fixed = format_conversations(tokenizer, conversations)
            inputs = tokenizer(
                convs_fmt_fixed, return_tensors="pt", padding=True, max_length=args.max_length, truncation=True
            ).to(device)
            logits = model(**inputs).logits.squeeze(-1).cpu()
            scores = logits.tolist()

        idx = 0
        for conv_len in conv_lengths:
            results.append(scores[idx : idx + conv_len])
            idx += conv_len

    output_path = args.output_dir / f"{args.output_prefix}_rank{rank}.jsonl"
    args.output_dir.mkdir(exist_ok=True, parents=True)
    ds_sample = ds_sample.add_column("reward", results)
    ds_sample.to_json(output_path)


if __name__ == "__main__":
    main()
