import math
import time
import os
import logging
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import regex as re
from tqdm.auto import tqdm

import pyarrow.compute as pc
from transformers import PreTrainedTokenizerFast
from datasets import Dataset, load_dataset, disable_caching

from sample_efficient_gpt.tokenizer.pretokenization import find_chunk_boundaries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
disable_caching()

if int(os.getenv("STAGE", "1")) == 2:
    COSMOPEDIA_THRESHOLD = 2.3
    FINEMATH_THRESHOLD = 4.1
    DCLM_EDU_THRESHOLD = 3.2
    STACK_EDU_THRESHOLD = 4.1
    FINEPDFS_THRESHOLD = 2.1
    PLEIAS_SYNTH_THRESHOLD = 1.85
    print("Using thresholds from stage 2")
else:
    COSMOPEDIA_THRESHOLD = 2.0
    FINEMATH_THRESHOLD = 3.7
    DCLM_EDU_THRESHOLD = 2.75
    STACK_EDU_THRESHOLD = 3.75
    FINEPDFS_THRESHOLD = 1.75
    PLEIAS_SYNTH_THRESHOLD = 1.65


class TokenizerProcessor:
    def __init__(self, tokenizer_name, special_tokens=[]):
        # Special tokens and split regex
        self.tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)
        self.special_tokens = special_tokens
        escaped = [re.escape(tok) for tok in sorted(self.special_tokens, reverse=True)]
        base_tok = "<|endoftext|>"
        self.split_re = f"({'|'.join(escaped)})" if escaped else f"({re.escape(base_tok)})"

    def encode_hf(self, filepath: str, tokenized_path, limit: int = -1):
        """
        Method to tokenize saved HF dataset directly as a form of parquet or jsonl files
        """
        path = Path(filepath)
        if limit > 0:
            token_count = limit
        else:
            token_count = None
        current_count = 0
        # assert path.suffix == ".parquet", "Only parquet files are supported"
        parquet_files = list(path.glob("**/*.parquet"))
        jsonl_files = list(path.glob("*.jsonl.gz"))
        if parquet_files:
            for chunk_path in tqdm(parquet_files):
                name = "".join(str(chunk_path.name).split(".")[:-1])
                out_path = (tokenized_path / name).with_suffix(".npy")
                if out_path.exists():
                    if limit > 0:
                        current_count += np.load(str(out_path), mmap_mode="r").shape[0]
                    logger.info(f"{str(out_path)} exists, skipping")
                    continue
                if token_count is not None and current_count > token_count:
                    logger.info("reached limit")
                    return
                # df = pd.read_parquet(str(chunk_path))
                # df = df[df["metadata"].apply(lambda x: x["edu_score"]) > 2.75]
                # df["text"] = df["text"].apply(lambda s: s + "<|endoftext|>")
                # ds = Dataset.from_pandas(df)

                logger.info("read parquet")
                ds = load_dataset("parquet", data_files=str(chunk_path), split="train")  # Arrow-backed, zero-copy-ish
                # for dclm-edu
                if "dclm-edu" in str(chunk_path):
                    ds = ds.filter(
                        lambda ex: ex["metadata"]["edu_score"] > DCLM_EDU_THRESHOLD, num_proc=4, load_from_cache_file=False
                    )
                # for finemath
                if "finemath" in str(chunk_path):
                    ds = ds.filter(lambda ex: ex["score"] > FINEMATH_THRESHOLD, num_proc=4, load_from_cache_file=False)
                if "cosmopedia-v2-textbook-7b" in str(chunk_path):
                    ds = ds.filter(lambda ex: ex["score"] > COSMOPEDIA_THRESHOLD, num_proc=4, load_from_cache_file=False)
                if "stack_edu_375plus_20B" in str(chunk_path):
                    ds = ds.filter(lambda ex: ex["score"] > STACK_EDU_THRESHOLD, num_proc=4, load_from_cache_file=False)
                if "finepdfs_en_2020plus_edu" in str(chunk_path):
                    ds = ds.filter(lambda ex: ex["score"] > FINEPDFS_THRESHOLD, num_proc=4, load_from_cache_file=False)
                if "pleias_synth" in str(chunk_path):
                    ds = ds.filter(lambda ex: ex["score"] > PLEIAS_SYNTH_THRESHOLD, num_proc=4, load_from_cache_file=False)

                if not "the_stack_v2_" in str(chunk_path):
                    ds = ds.map(
                        lambda ex: {"text": ex["text"] + "<|endoftext|>"}, num_proc=4, load_from_cache_file=False
                    )
                tokens, offsets = self._process_ds(ds)
                np.save(str(out_path), tokens)
                np.save(str((tokenized_path / f"offsets_{chunk_path.stem}").with_suffix(".npy")), offsets)
                current_count += tokens.shape[0]
                logger.info(f"current count: {current_count}")
        elif jsonl_files:
            # for marin-arxiv
            # chunk_size = 50
            # for everything else
            chunk_size = 100
            chunks = [jsonl_files[i : i + chunk_size] for i in range(0, len(jsonl_files), chunk_size)]
            for i, chunk in tqdm(list(enumerate(chunks))):
                if token_count is not None and current_count > token_count:
                    logger.info("reached limit")
                    break
                logger.info("read jsonl")
                try:
                    ds = load_dataset("json", data_files=[str(x) for x in chunk], split="train")
                except:
                    logger.info("error occured while reading the dataset, skipping")
                    continue
                ds = ds.map(lambda ex: {"text": ex["text"] + "<|endoftext|>"}, num_proc=4)
                tokens, offsets = self._process_ds(ds)
                np.save(str(tokenized_path / f"{i}.npy"), tokens)
                np.save(str(tokenized_path / f"offsets_{i}.npy"), offsets)
                current_count += tokens.shape[0]
                logger.info(f"current count: {current_count}")

    def encode_hf_txt(self, filepath: Path, tokenized_path, chunk_size):
        """Method to encode folder with txt files with HF tokenizer directly"""
        txt_files = list(filepath.glob("*val*.txt")) + list(filepath.glob("*train*.txt"))
        for txt_path in txt_files:
            all_tokens = None
            for i, chunk in enumerate(self._split_text_file(txt_path, chunk_size)):
                docs = self._process_text_chunk(chunk)
                ds = Dataset.from_dict({"text": docs})
                tokens, offsets = self._process_ds(ds)
                if all_tokens is None:
                    all_tokens = tokens
                else:
                    all_tokens = np.concat((all_tokens, tokens))
                if (i + 1) % 5 == 0:
                    np.save(str((tokenized_path / f"{txt_path.stem}").with_suffix(".npy")), all_tokens)
            np.save(str((tokenized_path / f"{txt_path.stem}").with_suffix(".npy")), all_tokens)

    def _process_ds(self, ds: Dataset):
        ds = ds.map(lambda x: {"tokens": self.tok.encode(x["text"])}, num_proc=24, load_from_cache_file=False)
        col = "tokens"

        # ChunkedArray of list<int> from the HF Dataset
        ca = ds.data.column(col)

        # Flatten values and get per-row lengths directly via Arrow
        flat_ca = pc.list_flatten(ca)  # ChunkedArray<int64>
        lens_ca = pc.list_value_length(ca)  # ChunkedArray<int32>

        flat = flat_ca.combine_chunks().to_numpy()  # 1D np.int64 of all tokens
        lens = lens_ca.combine_chunks().to_numpy()  # 1D lengths per row

        # (optional) safety check before casting to uint16
        if flat.size and flat.max() >= (1 << 16):
            raise ValueError("Token ids exceed uint16 range; choose uint32 instead.")

        flat = flat.astype(np.uint16, copy=False)

        # Build offsets so you can reconstruct rows: row_i = flat[offsets[i]:offsets[i+1]]
        offsets = np.empty(len(lens) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lens, out=offsets[1:])

        # Save
        return flat, offsets

    def _split_text_file(self, filepath, chunk_size: int = 32 * 1024 * 1024):
        path = Path(filepath)
        size = path.stat().st_size
        n_chunks = math.ceil(size / chunk_size)
        boundaries = find_chunk_boundaries(path.open("rb"), n_chunks, b" ")
        with path.open("rb") as f:
            for chunk_i, (start, end) in tqdm(
                enumerate(zip(boundaries[:-1], boundaries[1:])), total=len(boundaries) - 1
            ):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                yield chunk

    def _process_text_chunk(self, inp: str):
        docs = []
        for doc in re.splititer(self.split_re, inp, concurrent=True):
            if doc and doc in self.special_tokens:
                continue
            docs.append(doc + "<|endoftext|>")
        return docs

    def encode(self, inp: str) -> list[list[int], list[int]]:
        docs = self._process_text_chunk(inp)
        if len(docs) == 1:
            return self.tok.encode(docs[0]), [0]

        ds = Dataset.from_dict({"text": docs})
        return self._process_ds(ds)

        # ---- Later: load & reconstruct ith sequence ----
        # z = np.load("tokens_ragged_u16.npz")
        # values, offsets = z["values"], z["offsets"]
        # i_seq = values[offsets[i]:offsets[i+1]]

    def encode_file(self, filepath: str | Path, chunk_size: int = 1024 * 1024) -> list[int]:
        all_tokens = None
        for chunk in self._split_text_file(filepath):
            tokens, offsets = self.encode(chunk)
            if all_tokens is None:
                all_tokens = tokens
            else:
                all_tokens = np.concat((all_tokens, tokens))
        np.save(str((tokenized_path / f"{path.stem}").with_suffix(".npy")), all_tokens)

        # with path.open("rb") as f:
        # for chunk_i, (start, end) in tqdm(
        #     enumerate(zip(boundaries[:-1], boundaries[1:])), total=len(boundaries) - 1
        # ):
        #     f.seek(start)
        #     chunk = f.read(end - start).decode("utf-8", errors="ignore")
        #     tokens, offsets = self.encode(chunk)
        #     np.savez(
        #         str((tokenized_path / f"{path.stem}_{chunk_i}").with_suffix(".npz")), values=tokens, offsets=offsets
        #     )
        return tokens

    def decode(self, ids: int | bytes | list[int | bytes]) -> str:
        if not isinstance(ids, list):
            ids = [ids]
        b = b""
        for i in ids:
            if isinstance(i, int):
                b += self.vocab.get(i, b"")
            else:
                b += i
        return b.decode("utf-8", errors="replace")


def parse_args():
    p = ArgumentParser()
    # p.add_argument("--tokenizer-name", default="thepowerfuldeez/0925_bigbench_code_tokenizer")
    p.add_argument("--tokenizer-name", default="data_dclm_edu/tokenizer_superbpe_hf/")
    p.add_argument("--data-path", default="/mnt/harddrive/datasets/bigcode_the_stack_v2_updated_smol/")
    p.add_argument(
        "--tokenized-data-path", default="/mnt/harddrive/datasets/bigcode_the_stack_v2_updated_smol/tokenized_54770"
    )
    p.add_argument("--stats-name", default="64446_stats.json")
    p.add_argument("--include-val-data", default=0, type=int)
    p.add_argument("--is-hf", default=1, type=int)
    p.add_argument("--limit", default=-1, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tok = TokenizerProcessor(args.tokenizer_name, special_tokens=["<|endoftext|>"])
    input_dir = Path(args.data_path)
    limit = args.limit

    logger.info("start")
    tokenized_path = Path(args.tokenized_data_path)
    tokenized_path.mkdir(exist_ok=True, parents=True)

    t0 = time.monotonic()
    if args.is_hf and not list(input_dir.glob("*train*.txt")):
        print("tokenizing jsonl / parquet files by chunks with hf")
        tok.encode_hf(input_dir, tokenized_path, limit=limit)
    elif args.is_hf and list(input_dir.glob("*train*.txt")):
        print("tokenizing txt files with hf")
        tok.encode_hf_txt(input_dir, tokenized_path, chunk_size=128 * 1024 * 1024)
    else:
        compression_ratios = {}
        for f in input_dir.glob("*val*.txt"):
            # sample 100 docs from each dataset
            ratios = []
            for doc in re.split(tok.split_re, f.read_text())[:200]:
                if doc != "<|endoftext|>" and doc:
                    n_bytes = len(doc.encode())
                    # print(doc)
                    toks, _ = tok.encode(doc)
                    # print(toks)
                    n_tokens = len(toks)
                    ratio = n_bytes / n_tokens
                    ratios.append(ratio)
            avg_ratio = np.mean(ratios)
            logger.info(f"{f.name} compression ratio: {avg_ratio:.2f}")
            compression_ratios[f.name] = avg_ratio
        (input_dir / args.stats_name).write_text(json.dumps(compression_ratios))

        fpaths = list(input_dir.glob("*train.txt"))
        if args.include_val_data == 1:
            fpaths.extend(list(input_dir.glob("*val*.txt")))
        for fpath in fpaths:
            t0 = time.monotonic()
            tokens = tok.encode_file(fpath, chunk_size=128 * 1024 * 1024)
            taken = time.monotonic() - t0
            logger.info(f"Processed {str(fpath)}")
            logger.info(f"Took {taken:.1f} s.")
            logger.info(f"Throughput: {fpath.stat().st_size / (1024 * 1024) / taken:.2f} MB/s")
            fname = fpath.name
    taken = time.monotonic() - t0
    logger.info(f"Took {taken:.1f} s.")
