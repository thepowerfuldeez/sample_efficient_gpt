import math
import pickle
import time
import logging
import json
import heapq
from argparse import ArgumentParser
from pathlib import Path
from itertools import count
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import regex as re
from tqdm.auto import tqdm

from sample_efficient_gpt.tokenizer.pretokenization import find_chunk_boundaries

# PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
# upgraded pattern with digits grouping
PAT = re.compile(r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")


@dataclass()
class Token:
    value: int
    next = None
    prev = None
    version: int = 0

    def __repr__(self):
        if self.next is not None:
            n = self.next.value
        else:
            n = None
        if self.prev is not None:
            p = self.prev.value
        else:
            p = None
        if p is None and n is None:
            return "X"
        return f"Token({self.value}, v={self.version}, n={n}, p={p})"


class MinHeap:
    def __init__(self, bpe_ranks: dict[tuple[int, int], int]):
        self.heap = []
        heapq.heapify(self.heap)
        self.tie_breaker = count()
        self.bpe_ranks = bpe_ranks

    def push(self, left: Token):
        if left and left.next:
            pair = (left.value, left.next.value)
            # logger.info(f"check {pair}")
            if pair in self.bpe_ranks:
                rank = self.bpe_ranks[pair]
                tie_breaker = next(self.tie_breaker)
                # logger.info(f"push {(rank, tie_breaker, left.version, left, left.next)}")
                heapq.heappush(self.heap, (rank, tie_breaker, left.version, left))

    def pop(self):
        rank, _, version_at_push, node = heapq.heappop(self.heap)
        # logger.info(f"pop {rank=} {version_at_push=} {node=}")
        return rank, node, version_at_push

    def __bool__(self):
        return bool(self.heap)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def process(index_and_args) -> tuple[int, list[int]]:
#     index, args = index_and_args
#     (f, start, end, split_re, special_tokens, byte2int, PAT, merges, merge_key) = args
#     chunk: str = f.read(end - start).decode("utf-8")

#     res = []
#     for doc in re.splititer(split_re, chunk, concurrent=True):
#         # handle special tokens separately
#         if doc in special_tokens:
#             res.append(byte2int[bytes(doc.encode())])
#             continue
#         for tok in PAT.finditer(doc, concurrent=True):
#             tok_bytes = bytes(tok.group().encode())
#             key = []
#             for i in range(len(tok_bytes)):
#                 s = tok_bytes[i : i + 1]
#                 if s in byte2int:
#                     key.append(byte2int[s])
#                 else:
#                     key.extend(list([s]))
#             for new_id, (left, right) in merges:
#                 key, _ = merge_key(left, right, key, new_id)
#                 if len(key) == 1:
#                     break
#             res.extend(key)
#     return index, res


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None, superbpe_transition_idx: int | None = None):
        # Special tokens and split regex
        self.special_tokens = special_tokens or []
        # Vocabulary and reverse map
        self.vocab = vocab
        self.byte2int = {v: k for k, v in vocab.items()}
        self.lut256 = [-1] * 256
        for b in range(256):
            self.lut256[b] = self.byte2int[bytes([b])]

        # self.merges = [
        #     ((self.byte2int[left], self.byte2int[right]), self.byte2int[left + right]) for left, right in merges
        # ]

        escaped = [re.escape(tok) for tok in sorted(self.special_tokens, reverse=True)]
        base_tok = "<|endoftext|>"
        self.split_re = f"({'|'.join(escaped)})" if escaped else f"({re.escape(base_tok)})"
        # idx when we start encoding differently
        self.superbpe_transition_idx = superbpe_transition_idx
        if superbpe_transition_idx is not None:
            merges_superbpe = merges[superbpe_transition_idx:]
            self.merge_map_superbpe, self.bpe_ranks_superbpe = self._init_merge_map(merges_superbpe)
            merges = merges[:superbpe_transition_idx]

        self.merge_map, self.bpe_ranks = self._init_merge_map(merges)

    def _init_merge_map(self, merges):
        # Build merge_map: (left_id, right_id) -> new_id
        merge_map: dict[tuple[int, int], int] = {}
        bpe_ranks: dict[tuple[int, int], int] = {}
        for rank, (left_bytes, right_bytes) in enumerate(merges):
            # IDs for left and right tokens
            left_id = self.byte2int[left_bytes]
            right_id = self.byte2int[right_bytes]
            # combined bytes sequence should already be in vocab for BPE merges
            combined = left_bytes + right_bytes
            new_id = self.byte2int.get(combined)
            if new_id is None:
                # fallback: map concatenation not in vocab
                continue
            pair = (left_id, right_id)
            merge_map[pair] = new_id
            bpe_ranks[pair] = rank
        return merge_map, bpe_ranks

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
        superbpe_transition_idx: int | None = None,
    ):
        """Constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens"""
        return cls(
            vocab=pickle.loads(Path(vocab_filepath).read_bytes()),
            merges=pickle.loads(Path(merges_filepath).read_bytes()),
            special_tokens=special_tokens,
            superbpe_transition_idx=superbpe_transition_idx,
        )

    @lru_cache(maxsize=10_000_000)
    def _encode_bytes_cached(self, tok_bytes: bytes) -> tuple[int]:
        key = [self.lut256[b] for b in tok_bytes]
        return tuple(self._heap_merge(key))

    def _heap_merge(self, key: list[int | bytes], bpe_ranks=None, merge_map=None) -> list[int]:
        """
        1. Build a double linked list from input sequence
            Each node has value, version, next and prev
        2. Use MinHeap with tie breaker as itertools.count
        3. At every iteration, take a node with lowest bpe rank
            then validate it has right neighbor and collapse to new id
            Increase version of left node
            Relink left node to a new neighbor (left.next.next)
        4. Add 2 neighbors after merge back to the heap
        """
        if bpe_ranks is None:
            bpe_ranks = self.bpe_ranks
        if merge_map is None:
            merge_map = self.merge_map

        tokens = [Token(v) for v in key]
        heap = MinHeap(bpe_ranks=bpe_ranks)

        for i in range(len(tokens) - 1):
            tokens[i].next = tokens[i + 1]
            tokens[i + 1].prev = tokens[i]
        # logger.info("---------------")
        # logger.info(tokens)
        # logger.info([self.decode(n.value) for n in tokens])

        for n in tokens[:-1]:
            heap.push(n)

        while heap:
            rank_at_push, left, version_at_push = heap.pop()
            if left.version != version_at_push:
                # logger.info("version mismatch")
                continue
            right = left.next
            if not right:
                # logger.info("no right node")
                continue
            pair = (left.value, right.value)
            if pair not in bpe_ranks:
                # logger.info("incorrect pair")
                continue
            if bpe_ranks[pair] != rank_at_push:
                # logger.info(f"incorrect rank: {bpe_ranks[pair]=} {rank_at_push=}")
                continue

            new_id = merge_map[pair]
            left.value = new_id
            left.version += 1
            # logger.info(f"{pair} -> {new_id}")

            # unlink right (fully detach and invalidate)
            rn = right.next
            right.next = None
            right.prev = None
            right.version += 1

            # relink left
            left.next = rn
            if rn:
                rn.prev = left

            # logger.info(tokens)

            if left.prev:
                heap.push(left.prev)
            heap.push(left)
        out = []
        cur = tokens[0]
        # rollback to find actual head
        while cur and cur.prev:
            cur = cur.prev
        while cur:
            out.append(cur.value)
            cur = cur.next
        # logger.info(out)
        return out

    def encode(self, inp: str) -> list[int | bytes]:
        out: list[int | bytes] = []
        for doc in re.splititer(self.split_re, inp, concurrent=True):
            if doc and doc in self.special_tokens:
                token_bytes = doc.encode()
                out.append(self.byte2int[token_bytes])
                continue
            for i, tok in enumerate(PAT.finditer(doc, concurrent=True)):
                tok_bytes = tok.group().encode("utf-8")
                key = self._encode_bytes_cached(tok_bytes)
                out.extend(key)
            if self.superbpe_transition_idx is not None:
                out = self._heap_merge(out, bpe_ranks=self.bpe_ranks_superbpe, merge_map=self.merge_map_superbpe)
        return out

    def encode_iterable(self, iterable):
        for line in tqdm(iterable, desc="Lines processed: "):
            yield from self.encode(line)

    def encode_file(self, filepath: str | Path, chunk_size: int = 1024 * 1024) -> list[int]:
        path = Path(filepath)
        size = path.stat().st_size
        n_chunks = math.ceil(size / chunk_size)
        boundaries = find_chunk_boundaries(path.open("rb"), n_chunks, b" ")
        tokens: list[int] = []
        with path.open("rb") as f:
            for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), total=len(boundaries) - 1):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                tokens.extend(self.encode(chunk))
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
    p.add_argument("--vocab-path")
    p.add_argument("--merges-path")
    p.add_argument("--data-path", default="/mnt/harddrive/datasets/bigcode_the_stack_v2_updated_smol/")
    p.add_argument(
        "--tokenized-data-path", default="/mnt/harddrive/datasets/bigcode_the_stack_v2_updated_smol/tokenized_54770"
    )
    p.add_argument("--superbpe-transition-idx", default=None, type=int)
    p.add_argument("--stats-name", default="64446_stats.json")
    p.add_argument("--include-val-data", default=0, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tok = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        superbpe_transition_idx=args.superbpe_transition_idx,
        special_tokens=["<|endoftext|>"],
    )
    input_dir = Path(args.data_path)

    compression_ratios = {}
    for f in input_dir.glob("*val*.txt"):
        # sample 100 docs from each dataset
        ratios = []
        for doc in re.split(tok.split_re, f.read_text())[:200]:
            if doc != "<|endoftext|>" and doc:
                n_bytes = len(doc.encode())
                # print(doc)
                toks = tok.encode(doc)
                # print(toks)
                n_tokens = len(toks)
                ratio = n_bytes / n_tokens
                ratios.append(ratio)
        avg_ratio = np.mean(ratios)
        print(f"{f.name} compression ratio: {avg_ratio:.2f}")
        compression_ratios[f.name] = avg_ratio
    (input_dir / args.stats_name).write_text(json.dumps(compression_ratios))

    tokenized_path = Path(args.tokenized_data_path)
    tokenized_path.mkdir(exist_ok=True, parents=True)

    fpaths = list(input_dir.glob("*train*.txt"))
    if args.include_val_data == 1:
        fpaths.extend(list(input_dir.glob("*val*.txt")))
    for fpath in fpaths:
        t0 = time.monotonic()
        tokens = tok.encode_file(fpath, chunk_size=8 * 1024 * 1024)
        taken = time.monotonic() - t0
        logger.info(f"Processed {str(fpath)}")
        logger.info(f"Took {taken:.1f} s.")
        logger.info(f"Throughput: {fpath.stat().st_size / (1024 * 1024) / taken:.2f} MB/s")
        fname = fpath.name
        np.save(str((tokenized_path / fname).with_suffix(".npy")), np.array(tokens, dtype="uint16"))
