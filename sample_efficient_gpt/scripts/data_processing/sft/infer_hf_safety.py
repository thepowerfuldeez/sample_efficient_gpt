"""Safety classifier. Shard across ranks when launched with torchrun."""

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
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run safety classifier with optional torchrun sharding.")
    parser.add_argument("--data-file", required=True, help="JSONL dataset with generated outputs.")
    parser.add_argument(
        "--model-checkpoint",
        default="Jazhyc/modernbert-wildguardmix-classifier",
        help="Safety classifier checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Rows to score per forward pass.")
    parser.add_argument("--max-length", type=int, default=2048, help="Max tokens per sample.")
    parser.add_argument("--world-size", type=int, default=None, help="Override world size (else torchrun env).")
    parser.add_argument("--rank", type=int, default=None, help="Override rank (else LOCAL_RANK/WORLD_SIZE).")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="k_safety",
        help="Prefix for rank-suffixed JSONL outputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for outputs.")
    args = parser.parse_args()

    rank = args.rank if args.rank is not None else int(os.environ.get("LOCAL_RANK", 0))
    world_size = args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", 1))

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    model, tokenizer = get_model(device, args.model_checkpoint)
    classes = ["safe", "harmful"]

    ds = load_dataset("json", data_files=args.data_file)["train"]
    total = len(ds)
    per_rank = (total + world_size - 1) // world_size
    start = rank * per_rank
    end = min(total, start + per_rank)
    if start >= total:
        raise ValueError(f"Rank {rank} has no data to process (start {start} >= total {total}).")
    ds_sample = ds.select(range(start, end))

    results = []
    for i in tqdm(range(0, len(ds_sample), args.batch_size)):
        batch = ds_sample.select(range(i, min(len(ds_sample), i + args.batch_size)))
        text_batch = []
        for row in batch:
            inp = row["input"]["messages"][-1]["content"]
            candidates = [row["output"]] + row["generated_outputs"]
            text_batch.extend([inp + "\n\n" + cand for cand in candidates])

        with torch.autocast("cuda", torch.bfloat16) if torch.cuda.is_available() else torch.autocast(
            "cpu", torch.bfloat16
        ):
            inputs = tokenizer(text_batch, return_tensors="pt", padding=True, max_length=args.max_length, truncation=True)
            for k, v in inputs.items():
                inputs[k] = inputs[k].to(device)
            preds = torch.argmax(model(**inputs).logits, dim=-1).cpu()
        sample_classes = [classes[j] for j in preds]

        # Group back per prompt
        idx = 0
        grouped = []
        for row in batch:
            num_candidates = 1 + len(row["generated_outputs"])
            grouped.append(sample_classes[idx : idx + num_candidates])
            idx += num_candidates
        results.extend(grouped)

    output_path = args.output_dir / f"{args.output_prefix}_rank{rank}.jsonl"
    args.output_dir.mkdir(exist_ok=True, parents=True)
    ds_sample = ds_sample.add_column("safety_class", results)
    ds_sample.to_json(output_path)


if __name__ == "__main__":
    main()
