"""Module with pretokenization utils. Can be executed for speed benchmark"""

import time
import os
from typing import BinaryIO
from pathlib import Path

from pathos.multiprocessing import ThreadingPool as Pool

from fastsplit import Splitter


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize(
    splitter: Splitter, filepath: str, num_processes: int = 1, special_token: str = "<|endoftext|>"
) -> dict[str, int]:
    """
    Run pre-tokenization algorithm.

    1. Splits file by equally sized chunks separated by EOS
    2. Each chunk is processed by split function in parallel
    """
    handle: BinaryIO = Path(filepath).open("rb")
    boundaries = find_chunk_boundaries(handle, num_processes * 4, special_token.encode())
    handle.close()

    args = [(filepath, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    pre_token_counts: dict[str, int] = {}
    if num_processes > 1:
        with Pool(num_processes) as p:
            results = p.map(splitter.seek_and_split, *zip(*args))
        for pre_token_counts_sample in results:
            for k, v in pre_token_counts_sample.items():
                if k in pre_token_counts:
                    pre_token_counts[k] += v
                else:
                    pre_token_counts[k] = v
    else:
        for fp, s, e in args:
            # returns counts automatically
            pre_token_counts_sample = splitter.seek_and_split(fp, s, e)
            for k, v in pre_token_counts_sample.items():
                if k in pre_token_counts:
                    pre_token_counts[k] += v
                else:
                    pre_token_counts[k] = v

    return pre_token_counts


if __name__ == "__main__":
    splitter = Splitter("<|endoftext|>")
    # filepath = "tests/fixtures/tinystories_sample_5M.txt"
    # filepath = "data/TinyStoriesV2-GPT4-train.txt"
    filepath = "data/TinyStoriesV2-GPT4-mid3.txt"
    t0 = time.monotonic()
    pre_token_counts = pre_tokenize(splitter, filepath, num_processes=1)
    t1 = time.monotonic()
    print(f"Took {t1 - t0:.2f}s.")
