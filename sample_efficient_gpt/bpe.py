import logging
import os
import math
import pickle
import time
import random
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict

import regex as re
import numpy as np
from tqdm.auto import tqdm

from sample_efficient_gpt.pretokenization import Splitter, pre_tokenize, find_chunk_boundaries

logging.basicConfig(level=os.getenv("LOGLEVEL", logging.INFO))
logger = logging.getLogger(__name__)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(PAT)


def pack_u16_np(view: np.ndarray) -> bytes:
    # Treat input as unsigned 16-bit; avoid copies when possible
    a = np.asarray(view, dtype=np.uint16)  # no copy if dtype already u16/i16
    if not a.flags.c_contiguous:  # contiguous makes packing fast
        a = np.ascontiguousarray(a)
    # normalize to little-endian bytes so keys are stable across platforms
    if a.dtype.byteorder not in ("<", "="):
        a = a.byteswap().newbyteorder("<")
    return a.tobytes()


def unpack_u16_np(b: bytes) -> np.ndarray:
    # Zero-copy read-only NumPy view over the bytes
    return np.frombuffer(b, dtype=np.uint16)


def unpack_u16(b):
    # zero-copy view:
    return memoryview(b).cast("H")  # returns a read-only sequence-like view


class BPE:
    def __init__(self, special_tokens: list[str], vocab_size: int = 355, save_every: int = 10000, save_dir: str = "."):
        self.vocab_size: int = vocab_size

        self.merges: list[tuple[bytes | int, bytes | int]] = []
        self.merges_tuples: list[tuple[bytes, bytes]] = []
        self.special_tokens = special_tokens
        self.special_tokens_bytes = [x.encode() for x in special_tokens]

        self.splitter = Splitter("<|endoftext|>")
        self.split_re = "(" + "|".join([re.escape(tok) for tok in self.special_tokens]) + ")"

        self.vocab: dict[int, bytes] = {256 + i: special_tok for i, special_tok in enumerate(self.special_tokens_bytes)}
        self.new_id_to_bytes: dict[int, int | bytes] = self.vocab.copy()
        for i in range(256):
            self.vocab[i] = bytes([i])

        self.sort_time = 0
        self.second_best_key = None

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.save_every = save_every
        self.vocab_name, self.merges_name = "vocab.pickle", "merges.pickle"

    @property
    def cur_vocab_size(self):
        """Current vocab size (different from self.vocab_size which is target vocab size)"""
        return len(self.vocab)

    def break_ties(self, sorted_all_counts):
        """
        This can be replaced with single sort_key function that is passed as a key to sorted
        but it would be a bit slower
        """
        _, max_cnt = sorted_all_counts[0]
        ties = []
        candidates = []
        for i in range(len(sorted_all_counts)):
            key, count = sorted_all_counts[i]
            if count == max_cnt:
                ties.append(tuple(self.vocab[id_] for id_ in key))
                candidates.append(key)
            else:
                break
        best_from_ties = max(ties)
        for i, cand in enumerate(candidates):
            if ties[i] == best_from_ties:
                max_key = cand
                break
        return max_key

    def convert(self, entry: bytes | int) -> bytes:
        """
        Convert tokens
        """
        if entry in self.vocab:
            return self.vocab[entry]
        bytestr: list[bytes] = self.new_id_to_bytes.get(entry, [entry])
        while any(elem in self.new_id_to_bytes for elem in bytestr):
            i = 0
            while i < len(bytestr):
                if bytestr[i] in self.new_id_to_bytes:
                    bytestr = bytestr[:i] + self.new_id_to_bytes[bytestr[i]] + bytestr[i + 1 :]
                i += 1
        else:
            return bytes(bytestr)

    def decode(self, entry: bytes | int | list[bytes | int]) -> str:
        """
        Returns list of bytes based on the mapping and converts to str

        We need to iteratively expand new tokens so that they are in 0-255 range
        We do that in a while loop and replace initial bytestring with bytes from mapping
        """
        if isinstance(entry, list) or isinstance(entry, tuple):
            return "".join([self.convert(e).decode("utf-8", errors="replace") for e in entry])
        return self.convert(entry).decode("utf-8", errors="replace")

    def encode(self, inp: str) -> list[int | bytes]:
        """
        Iteratively apply merges from self.merges to convert a string (sequence of bytes)
        into an encoded sequence
        """
        splitted_by_doc = re.split(self.split_re, inp)
        res = []
        for doc in splitted_by_doc:
            if doc in self.special_tokens:
                res.append(256 + self.special_tokens.index(doc))
                continue
            for tok in PAT.finditer(doc, concurrent=True):
                key = list(tok.group().encode())
                for new_id, (left, right) in enumerate(self.merges, 256 + len(self.special_tokens)):
                    key, _ = self.merge_key(left, right, key, new_id)
                    if len(key) == 1:
                        break
                res.extend(key)
        return res

    def encode_file(self, inp: str | Path) -> list[int | bytes]:
        file_path = Path(inp)
        chunk_size_in_bytes = 1024 * 1024 * 32
        n_chunks = math.ceil(file_path.stat().st_size / chunk_size_in_bytes)
        with Path(inp).open("rb") as f:
            boundaries = find_chunk_boundaries(f, n_chunks, b" ")
            f.seek(0)
            tokens = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk: str = f.read(end - start).decode("utf-8")
                tokens.extend(self.encode(chunk))
        return tokens

    def update_counts(
        self,
        pre_token_byte_counts: dict[tuple, int],
        pair_to_pre_tokens: dict[tuple, set] | None,
        all_counts: dict | None = None,
    ) -> dict[tuple[bytes], int]:
        """
        Update counts of all_counts by v with each pair of k
        Also update pair_to_pre_tokens if passed
        """
        if all_counts is None:
            all_counts: dict[tuple[bytes], int] = defaultdict(int)
        else:
            all_counts = defaultdict(int, all_counts)
        for k, v in pre_token_byte_counts.items():
            for i in range(len(k) - 1):
                key = (k[i], k[i + 1])
                all_counts[key] += v
                if pair_to_pre_tokens is not None:
                    pair_to_pre_tokens.setdefault(key, set()).add(k)
        return all_counts

    def iter_merge_cached(
        self,
        pre_token_byte_counts: dict[tuple[bytes], int],
        updated_keys=None,
        all_counts=None,
        pair_to_pre_tokens=None,
        all_updated_pairs=None,
    ) -> tuple[tuple[tuple[bytes], int], dict, set, dict, dict, set]:
        """
        More efficient implementation of iter_merge, that works 4x faster on 1000 iters (1.7s vs 6.3s)

        We operate on extra knowledge that we are only interested in updated pairs in iteration i-1
        at iteration i

        1. If we don't pass cached updated_keys, all_counts, pair_to_pre_tokens then algo is the same as iter_merge
        2. Instead, we just iterate over provided updated_keys (instead of all pre-tokens) and update provided
        all_counts
        3. Find rough estimation of pre-tokens by finding intersection with most frequent pair
        obtaining affected_pre_tokens
        4. Update pre_token_byte_counts more smartly -- only operating across affected_pre_tokens, where likely
        pre-tokens would change after the merge_key function
        5. Subtract count from every pair in a key that gets updated, and keep pair_to_pre_tokens updated
        At next iteration, only required keys will be updated and all of their pairs will be used
        """
        if pair_to_pre_tokens is None:
            # pair -> count
            pair_to_pre_tokens = {}
            all_counts: dict[tuple[bytes | int], int] = self.update_counts(pre_token_byte_counts, pair_to_pre_tokens)
            all_updated_pairs = set(all_counts.keys())
        else:
            # all_counts, pair_to_pre_tokens, pre_tokens_to_pairs, all_updated_pairs are restored from args
            all_counts: dict[tuple[bytes], int] = self.update_counts(
                {k: pre_token_byte_counts[k] for k in updated_keys},
                pair_to_pre_tokens,
                all_counts=all_counts,
            )

        # identify the most frequent pair
        tx = time.monotonic()

        # Optimized sorting algorithm -- keep checking only updated pairs + all previous sorted subset
        if self.second_best_key is not None and all_updated_pairs:
            sorted_subset = sorted(
                [
                    (k, all_counts[k])
                    for k in set([k for (k, v) in self.second_best_key]).union(all_updated_pairs)
                    if k in all_counts
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            self.second_best_key = sorted_subset
            max_key: tuple[bytes] = self.break_ties(sorted_subset)
        else:
            sorted_all_counts = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
            # We keep top10% of current iteration and re-use only that in next iter during sorting
            # This shaves time dramatically
            count_to_keep = math.ceil(len(sorted_all_counts) * 0.10)
            self.second_best_key = sorted_all_counts[:count_to_keep]
            max_key: tuple[bytes] = self.break_ties(sorted_all_counts)

        tx1 = time.monotonic()
        self.sort_time += tx1 - tx
        new_id = self.cur_vocab_size
        # logger.info(f"max freq is {all_counts[max_key]=}, max_key={self.decode(max_key)}")

        affected_pre_tokens: set[tuple[bytes]] = set()
        for (left, right), keys in pair_to_pre_tokens.items():
            if left in max_key and right in max_key:
                affected_pre_tokens.update(keys)

        new_pre_token_byte_counts = pre_token_byte_counts.copy()
        new_updated_keys: set[tuple[bytes]] = set()

        all_counts_updated = all_counts.copy()
        pair_to_pre_tokens_updated = pair_to_pre_tokens.copy()
        all_updated_pairs_updated = set()

        left, right = max_key
        for k in affected_pre_tokens:
            new_k, updated_indices = self.merge_key(left, right, k, new_id)
            if updated_indices:
                v = new_pre_token_byte_counts.pop(k)
                for pair in zip(k[:-1], k[1:]):
                    all_counts_updated[tuple(pair)] -= v
                    assert all_counts_updated[tuple(pair)] >= 0
                    pair_to_pre_tokens_updated[tuple(pair)].discard(k)

                # counts are not changed, we just need to re-write with a new key
                new_pre_token_byte_counts[new_k] = v
                new_updated_keys.add(new_k)

                for j in updated_indices:
                    if j > 0:
                        all_updated_pairs_updated.add((new_k[j - 1], new_k[j]))
                    if j < len(new_k) - 1:
                        all_updated_pairs_updated.add((new_k[j], new_k[j + 1]))

        return (
            (max_key, new_id),
            new_pre_token_byte_counts,
            new_updated_keys,
            {k: v for k, v in all_counts_updated.items() if v > 0},
            pair_to_pre_tokens_updated,
            all_updated_pairs_updated,
        )

    def iter_merge(
        self,
        pre_token_byte_counts: dict[tuple[bytes], int],
    ) -> tuple[tuple[tuple[bytes], int], dict]:
        """
        More efficient implementation of iter_merge, that works 4x faster on 1000 iters (1.7s vs 6.3s)

        We operate on extra knowledge that we are only interested in updated pairs in iteration i-1
        at iteration i

        1. If we don't pass cached updated_keys, all_counts, pair_to_pre_tokens then algo is the same as iter_merge
        2. Instead, we just iterate over provided updated_keys (instead of all pre-tokens) and update provided
        all_counts
        3. Find rough estimation of pre-tokens by finding intersection with most frequent pair
        obtaining affected_pre_tokens
        4. Update pre_token_byte_counts more smartly -- only operating across affected_pre_tokens, where likely
        pre-tokens would change after the merge_key function
        5. Subtract count from every pair in a key that gets updated, and keep pair_to_pre_tokens updated
        At next iteration, only required keys will be updated and all of their pairs will be used
        """
        # pair -> count
        all_counts: dict[tuple[bytes], int] = defaultdict(int)
        for k, v in pre_token_byte_counts.items():
            for i in range(len(k) - 1):
                all_counts[(k[i], k[i + 1])] += v

        # identify the most frequent pair
        tx = time.monotonic()

        sorted_all_counts = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
        max_key = self.break_ties(sorted_all_counts)

        tx1 = time.monotonic()
        self.sort_time += tx1 - tx
        if (tx1 - tx) > 0.01:
            logger.info(f"sort time is too long, took {tx1 - tx:.02f} s.")
        new_id = self.cur_vocab_size

        new_pre_token_byte_counts = pre_token_byte_counts.copy()

        left, right = max_key
        for k, v in pre_token_byte_counts.items():
            new_k, updated = self.merge_key(left, right, k, new_id)
            if updated:
                v = new_pre_token_byte_counts.pop(k)
                # counts are not changed, we just need to re-write with a new key
                new_pre_token_byte_counts[new_k] = v

        return (
            (max_key, new_id),
            new_pre_token_byte_counts,
        )

    def merge_key(self, left: int | bytes, right: int | bytes, k: tuple, new_id: int) -> tuple[tuple[bytes], list]:
        """
        Merges a pair of int | bytes into a key and returns a new key
        Example: a1, a2 = (111, 257)
                 k = (100, 111, 257)
                 new_id = 258 -> (100, 258)

        Returns: tuple with updated bytes
        """
        new_k = list(k)
        i = 0
        # updated indices in the modified key
        updated_indices: list[int] = []

        while i < len(new_k) - 1:
            if new_k[i] == left and new_k[i + 1] == right:
                new_k[i] = new_id
                new_k = new_k[: i + 1] + new_k[i + 2 :]
                updated_indices.append(i)
            i += 1
        return tuple(new_k), updated_indices

    def train(self, filepath: str, num_processes: int = 1):
        logger.info("Starting to train BPE")
        t0 = time.monotonic()
        pre_token_counts = pre_tokenize(self.splitter, str(filepath), num_processes=num_processes)

        t1 = time.monotonic()
        logger.info(f"Pre-tokenization finished in {t1 - t0:.1f} s.")
        self.pre_token_byte_counts: dict[tuple[bytes], int] = {
            tuple(k.encode()): v for k, v in pre_token_counts.items()
        }
        # TODO: if we are resuming, now pre token byte counts becomes already tokenized **documents** hashed as dict.
        # It is also helpful to remove duplicates or near duplicates that way, since we assume document is unique

        n_iters = max(0, self.vocab_size - self.cur_vocab_size)
        logger.info(f"Using {n_iters=}")

        # Non-efficient implementation
        # for i in range(n_iters):
        #     (updated_key, new_id), self.pre_token_byte_counts = self.iter_merge(self.pre_token_byte_counts)
        #     self.new_id_to_bytes[new_id] = updated_key
        #     v = self.convert(new_id)
        #     self.merges.append(updated_key)
        #     converted = (self.convert(updated_key[0]), self.convert(updated_key[1]))
        #     merges_tuples.append(converted)
        #     self.vocab[new_id] = v
        #     logger.info(f"iter: {i}, updated new id mapping with {new_id=}, {v=}")

        # cached, more efficient version
        updated_keys, all_counts, pair_to_pre_tokens, all_updated_pairs = None, None, None, None
        bar = tqdm(total=n_iters, desc="Training BPE")
        for i in range(n_iters):
            (
                (updated_key, new_id),
                self.pre_token_byte_counts,
                new_updated_keys,
                all_counts_updated,
                pair_to_pre_tokens_updated,
                all_updated_pairs_updated,
            ) = self.iter_merge_cached(
                self.pre_token_byte_counts,
                updated_keys,
                all_counts,
                pair_to_pre_tokens,
                all_updated_pairs,
            )
            # updated key is tuple[bytes] (left, right), represented as int
            self.new_id_to_bytes[new_id] = updated_key
            v = self.convert(new_id)
            self.merges.append(updated_key)
            converted = (self.convert(updated_key[0]), self.convert(updated_key[1]))
            # logger.info(f"merges at step {i}: {converted}")

            # merges tuples is now tuple of bytestrings
            self.merges_tuples.append(converted)
            self.vocab[new_id] = v
            # logger.info(f"iter: {i}, updated new id mapping with {new_id=}, {v=}")

            updated_keys, all_counts, pair_to_pre_tokens, all_updated_pairs = (
                new_updated_keys,
                all_counts_updated,
                pair_to_pre_tokens_updated,
                all_updated_pairs_updated,
            )

            if i % self.save_every == 0:
                self.save(iter=i)
                logger.info(f"Saved intermediate tokenizer at iter {i}")
            bar.update()
        t2 = time.monotonic()
        logger.info(f"Finished training in {t2 - t0:.1f} s.\nAverage iter time: {(t1 - t0) / n_iters:.5f} s.")
        logger.info(f"Total sort time was {self.sort_time:.2f} s.")
        return self.vocab, self.merges_tuples

    def pre_tokenize_resume(self, filepath: str, vocab, superbpe: bool = True):
        assert Path(filepath).suffix == ".npy", "Resuming supported only from .npy files!"
        out_arr = np.load(filepath, mmap_mode="r")
        pre_token_counts = {}
        document_boundaries = (out_arr == 256).nonzero()[0]

        # Prepare word boundaries either from using tokens with whitespace, or by custom rules in SuperBPE
        if superbpe:

            def process_diffs(diffs, word_boundaries):
                to_add = []
                for i, d in enumerate(diffs):
                    if d >= 20:
                        n_additions = min(d // 10, 11)
                        # corresponds to i, i+1 in word_boundaries
                        left = word_boundaries[i]
                        to_add.extend([left + 10 * i for i in range(1, n_additions + 1)])
                return np.array(to_add)

            # document + EOS tokens
            whitespace_tokens = np.array([i for i, v in vocab.items() if b":" in v] + [256])

            punctuation_tokens = np.array([i for i, v in vocab.items() if b":" in v or b"." in v or b"," in v])
            pre_punctuation_boundaries = np.isin(out_arr, punctuation_tokens).nonzero()[0] - 1

            word_boundaries = np.isin(out_arr, whitespace_tokens).nonzero()[0]

            # filter out pre punctuation boundaries that are covered by word boundaries
            pre_punctuation_boundaries = pre_punctuation_boundaries[
                ~np.isin(pre_punctuation_boundaries, word_boundaries)
            ]
            word_boundaries = np.concatenate((word_boundaries, pre_punctuation_boundaries))
            word_boundaries.sort(kind="mergesort")

            # filter by length
            diffs = word_boundaries[1:] - word_boundaries[:-1]
            max_length = int(np.percentile(diffs, 99))

            diff_boundaries_to_add = process_diffs(diffs, word_boundaries)
            diff_boundaries_to_add = diff_boundaries_to_add[~np.isin(diff_boundaries_to_add, word_boundaries)]
            word_boundaries = np.concatenate((word_boundaries, diff_boundaries_to_add))
            word_boundaries.sort(kind="mergesort")
        else:
            # document + EOS tokens
            whitespace_tokens = np.array([i for i, v in vocab.items() if v.startswith(b" ")] + [256])
            word_boundaries = np.isin(out_arr, whitespace_tokens).nonzero()[0]
        # which indices are actually EOS boundaries across all boundaries?
        document_boundaries_indices = np.isin(word_boundaries, document_boundaries).nonzero()[0]
        assert len(document_boundaries_indices) == len(document_boundaries)

        # when splitting by words we need to keep track of the document pointer, in which case
        # we should skip EOS token from adding as a key
        doc_boundary_ptr = 0
        pre_token_counts = {}
        should_change_left = False
        for i, right in tqdm(enumerate(word_boundaries), total=len(word_boundaries)):
            if i == 0:
                left = 0
            else:
                if i == document_boundaries_indices[doc_boundary_ptr]:
                    # we should skip EOS at the next iteration
                    should_change_left = True
                    doc_boundary_ptr += 1
            # we really don't need 1 token keys anymore
            # For SuperBPE: just take 10% of data due to OOM
            if right - left > 1 and (random.random() < 0.1 if superbpe else True):
                if superbpe:
                    length = right - left
                    # truncate by length
                    delta_length = max(0, length - max_length)
                    k = tuple(out_arr[left : right - delta_length].tolist())
                else:
                    k = tuple(out_arr[left:right].tolist())
                if k not in pre_token_counts:
                    pre_token_counts[k] = 1
                else:
                    pre_token_counts[k] += 1
            if should_change_left:
                left = right + 1
                should_change_left = False
            else:
                left = right
        assert all(256 not in k for k in pre_token_counts)
        return pre_token_counts

    def resume_train(
        self, filepath: str, new_vocab_size: int, vocab_path: str, merges_path: str, superbpe: bool = True
    ):
        """
        In this method, we are expecting already tokenized npy file for train
        """
        logger.info("Starting to train BPE")
        t0 = time.monotonic()

        self.vocab = pickle.loads(Path(vocab_path).read_bytes())
        bytes2new_id = {v: k for k, v in self.vocab.items()}

        self.merges_tuples = pickle.loads(Path(merges_path).read_bytes())
        # left, right
        self.merges = [(bytes2new_id[a], bytes2new_id[b]) for a, b in self.merges_tuples]

        self.new_id_to_bytes: dict[int, int | bytes] = {257 + i: v for i, v in enumerate(self.merges)}

        iter = len(self.vocab)
        logger.info(f"Resuming from {iter=}")
        self.vocab_size = new_vocab_size
        assert new_vocab_size > iter, "new vocab size must be greater than current vocab size!"

        self.pre_token_byte_counts = self.pre_tokenize_resume(filepath, self.vocab, superbpe=superbpe)
        # TODO: if we are resuming, now pre token byte counts becomes already tokenized **documents** hashed as dict.
        # It is also helpful to remove duplicates or near duplicates that way, since we assume document is unique

        n_iters = self.vocab_size - self.cur_vocab_size
        logger.info(f"Using {n_iters=}")

        # cached, more efficient version
        updated_keys, all_counts, pair_to_pre_tokens, all_updated_pairs = None, None, None, None
        bar = tqdm(total=n_iters, desc="Training BPE")
        for i in range(n_iters):
            (
                (updated_key, new_id),
                self.pre_token_byte_counts,
                new_updated_keys,
                all_counts_updated,
                pair_to_pre_tokens_updated,
                all_updated_pairs_updated,
            ) = self.iter_merge_cached(
                self.pre_token_byte_counts,
                updated_keys,
                all_counts,
                pair_to_pre_tokens,
                all_updated_pairs,
            )
            self.new_id_to_bytes[new_id] = updated_key
            v = self.convert(new_id)
            self.merges.append(updated_key)
            converted: tuple[bytes, bytes] = (self.convert(updated_key[0]), self.convert(updated_key[1]))
            # logger.info(f"merges at step {i}: {converted}")
            self.merges_tuples.append(converted)
            self.vocab[new_id] = v
            # logger.info(f"iter: {i}, updated new id mapping with {new_id=}, {v=}")

            updated_keys, all_counts, pair_to_pre_tokens, all_updated_pairs = (
                new_updated_keys,
                all_counts_updated,
                pair_to_pre_tokens_updated,
                all_updated_pairs_updated,
            )

            if i % self.save_every == 0:
                self.save(iter=i)
                logger.info(f"Saved intermediate tokenizer at iter {i}")
            bar.update()
        t2 = time.monotonic()
        logger.info(f"Finished training in {t2 - t0:.1f} s.")
        logger.info(f"Total sort time was {self.sort_time:.2f} s.")
        return self.vocab, self.merges_tuples

    def save(self, iter: int | None = None):
        if iter:
            vocab_path, merges_path = (
                self.save_dir / f"{iter}_{self.vocab_name}",
                self.save_dir / f"{iter}_{self.merges_name}",
            )
        else:
            vocab_path, merges_path = self.save_dir / self.vocab_name, self.save_dir / self.merges_name
        vocab_path.write_bytes(pickle.dumps(self.vocab))
        merges_path.write_bytes(pickle.dumps(self.merges_tuples))


def parse_args():
    p = ArgumentParser()
    p.add_argument("--data-path", default="data/owt_train.txt")
    # filepath = "data/TinyStoriesV2-GPT4-train.txt"
    # filepath = "data/TinyStoriesV2-GPT4-mid3.txt"
    # filepath = "data/TinyStoriesV2-GPT4-valid.txt"
    # filepath = "sample_efficient_gpt/test.txt"
    # filepath = "tests/fixtures/tinystories_sample_5M.txt"
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--num-processes", type=int, default=8)
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--save-dir", default=".")
    p.add_argument("--resume-from-vocab", default="")
    p.add_argument("--resume-from-merges", default="")
    p.add_argument("--superbpe", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bpe = BPE(["<|endoftext|>"], vocab_size=args.vocab_size, save_every=args.save_every, save_dir=args.save_dir)
    # bpe = BPE(["<|endoftext|>"], vocab_size=10000)
    # bpe = BPE(["<|endoftext|>"], vocab_size=1000)

    if args.resume_from_vocab and args.resume_from_merges:
        vocab, merges = bpe.resume_train(
            args.data_path,
            new_vocab_size=args.vocab_size,
            vocab_path=args.resume_from_vocab,
            merges_path=args.resume_from_merges,
            superbpe=args.superbpe == 1,
        )
    else:
        vocab, merges = bpe.train(args.data_path, num_processes=args.num_processes)

    # logger.info([bpe.decode(x) for x in list(vocab)[256:356]])
    # toks = bpe.encode("newest is a newest")
    # logger.info(toks)
    # logger.info([bpe.decode(tok) for tok in toks])

    bpe.save()
