"""Facility location selection over embeddings."""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from submodlib.functions.facilityLocation import FacilityLocationFunction
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Subset embeddings with Facility Location.")
    parser.add_argument("--data-file", required=True, help="JSONL file containing an 'embedding' column.")
    parser.add_argument("--start", type=int, default=0, help="Start index of slice to process.")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) of slice to process.")
    parser.add_argument("--select-frac", type=float, default=0.5, help="Fraction of examples to keep.")
    parser.add_argument("--output-file", type=Path, required=True, help="Where to write the selected subset.")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Embedding dimension.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Rows to load per chunk into the matrix.")
    return parser.parse_args()


def main():
    args = parse_args()
    ds = load_dataset("json", data_files=args.data_file)["train"]
    end = len(ds) if args.end is None else min(args.end, len(ds))
    ds = ds.select(range(args.start, end))
    print("total embs in slice", len(ds))

    n = len(ds)
    d = args.embedding_dim
    embeddings = np.empty((n, d), dtype=np.float16)

    for offset in tqdm(range(0, n, args.batch_size), desc="Building embedding matrix"):
        finish = min(offset + args.batch_size, n)
        batch = ds[offset:finish]["embedding"]
        embeddings[offset:finish] = np.asarray(batch, dtype=np.float16)

    number_select = int(args.select_frac * len(embeddings))

    obj = FacilityLocationFunction(
        n=len(embeddings),
        mode="dense",
        data=embeddings,
        metric="euclidean",
    )

    greedy_list = obj.maximize(
        budget=number_select,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    idx_list = [i for (i, _) in greedy_list]

    print(f"selected {len(idx_list)=}")
    ds_select = ds.select(idx_list)
    args.output_file.parent.mkdir(exist_ok=True, parents=True)
    ds_select.to_json(args.output_file)


if __name__ == "__main__":
    main()
