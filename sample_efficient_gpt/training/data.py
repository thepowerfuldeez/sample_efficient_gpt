import json
import threading
from pathlib import Path
from collections.abc import Iterator
from bisect import bisect

import torch
import torch.distributed as dist
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from jaxtyping import Int
from torch import Tensor

from sample_efficient_gpt.utils.logger import logger


class MemoryMappedDataset:
    def __init__(
        self,
        path_or_ds: str | Path | np.ndarray,
        context_length: int,
        device: str = "mps",
        seed: int | None = None,
        world_size: int = 1,
        rank: int = 0,
    ):
        """
        Reads numpy file in memory mapped mode
        Samples `context_length` chunks

        Can also accept a folder with .npy files (chunked dataset)

        Can also take only `rank` part of the batch for DDP training
        """
        self._prefetch_batch = None
        self._prefetch_event = threading.Event()
        self._prefetch_lock = threading.Lock()
        total_length = 0
        ds = []
        lengths = [0]
        path_or_ds = Path(path_or_ds)
        # JSON provided with exact amount of data to take
        if path_or_ds.suffix == ".json":
            path_mapping = json.loads(path_or_ds.read_text())
            logger.info(path_mapping)
            for path, token_count in path_mapping.items():
                arrays, read_lengths = self._read_folder(Path(path), lengths[-1], token_count)
                ds.extend(arrays)
                lengths.extend(read_lengths)
            total_length = lengths[-1]
        else:
            if path_or_ds.is_dir():
                arrays, read_lengths = self._read_folder(path_or_ds, lengths[-1])
                ds.extend(arrays)
                lengths.extend(read_lengths)
                total_length = lengths[-1]
            else:
                arr = self._read_file(path_or_ds)
                l = arr.shape[0]
                lengths.append(total_length)
                ds.append(arr)
                total_length += l
        self.total_length = total_length
        assert self.total_length > 0
        self.ds = ds
        self.lengths = lengths
        self.lengths_np = np.array(self.lengths, dtype=np.int64)
        self.chunk_sizes_np = np.array([arr.shape[0] for arr in self.ds], dtype=np.int64)

        # per-chunk window views for fast in-chunk slices
        self.chunk_windows = [sliding_window_view(arr, context_length + 1) for arr in self.ds]

        logger.info(f"Dataset length: {self.total_length}")
        self.context_length = context_length
        self.device = device
        if seed is not None:
            self.g = torch.Generator().manual_seed(seed)
        else:
            self.g = None

        self.world_size = world_size
        self.rank = rank
        self.is_distributed = world_size > 1
        self.local_batch_size = None
        self.N = self.total_length - self.context_length

        # Hack for GPUS which are uneven in their power
        self.gpu_names = None
        if self.is_distributed and torch.cuda.is_available():
            self.gpu_names = [torch.cuda.get_device_name(i) for i in range(self.world_size)]

    def _read_file(self, path):
        ds = np.load(path, mmap_mode="r")
        return ds

    def _read_folder(self, path: Path, offset_len: int, token_count: int | None = None):
        # dataset folder contains npy files
        arrays = []
        lengths = []
        current_length = 0
        for fp in sorted(path.glob("*.npy")):
            if token_count is not None and current_length > token_count:
                break
            if "offsets_" in fp.name:
                continue
            arr = self._read_file(fp)
            l = arr.shape[0]
            current_length += l
            lengths.append(offset_len + current_length)
            arrays.append(arr)
        return arrays, lengths

    def _split_indices(self, starts: np.ndarray):
        ds_left_chunks = np.searchsorted(self.lengths_np, starts, side="right") - 1
        # find ds chunks
        offs = starts - self.lengths_np[ds_left_chunks]
        context_span = self.context_length + 1
        # Determine which ones stay within the same chunk
        in_chunk = offs + context_span <= self.chunk_sizes_np[ds_left_chunks]
        return ds_left_chunks, offs, in_chunk

    def _gather_batch_numpy(self, starts: np.ndarray):
        left_chunks, offs, in_chunk = self._split_indices(starts)

        B = starts.shape[0]
        context_span = self.context_length + 1
        chunk = np.empty((B, context_span), dtype=np.int32)

        # Fast path: windows entirely within a single chunk (vectorized per unique chunk)
        for chunk_ind in np.unique(left_chunks[in_chunk]):
            mask = (left_chunks == chunk_ind) & in_chunk
            rows = np.nonzero(mask)[0]
            chunk[rows] = self.chunk_windows[chunk_ind][offs[rows]]  # (rows, ctx+1)

        # Slow path: windows crossing chunk boundary (rare): fill with two slices
        cross_rows = np.nonzero(~in_chunk)[0]
        for r in cross_rows:
            c = left_chunks[r]
            o = offs[r]
            left_cap = self.ds[c].shape[0] - o
            need = context_span
            take_left = min(left_cap, need)
            out = np.empty((context_span,), dtype=np.int32)
            out[:take_left] = self.ds[c][o : o + take_left].astype(np.int32, copy=False)

            rem = context_span - take_left
            cc = c + 1
            filled = take_left
            while rem > 0 and cc < len(self.ds):
                take = min(self.ds[cc].shape[0], rem)
                out[filled : filled + take] = self.ds[cc][:take].astype(np.int32, copy=False)
                rem -= take
                filled += take
                cc += 1
            chunk[r] = out

        inputs = chunk[:, :-1]
        targets = chunk[:, 1:]
        return inputs, targets

    def __len__(self):
        return self.N

    def get_iterator(self, batch_size: int) -> Iterator[tuple[Int[Tensor, "bs context"], Int[Tensor, "bs context"]]]:
        """
        Return an iterator of batches in a sequential order
        Last batch is dropped
        """
        context_span = self.context_length + 1
        sequential_indices = torch.arange(
            start=0, end=self.N - batch_size * context_span, step=batch_size * context_span
        )
        assert len(self.ds) == 1, "Only validation is supported (len(ds) == 1)"
        windows = self.chunk_windows[0]
        for i_start in sequential_indices:
            starts = torch.arange(i_start, i_start + batch_size * context_span, context_span).cpu().numpy()
            # bs x context_span
            chunk = windows[starts]
            batch_inputs = chunk[:, :-1].astype(np.int32, copy=False)
            batch_targets = chunk[:, 1:].astype(np.int32, copy=False)
            batch_inputs, batch_targets = torch.from_numpy(batch_inputs), torch.from_numpy(batch_targets)
            batch_inputs = batch_inputs.pin_memory()
            batch_targets = batch_targets.pin_memory()
            yield batch_inputs, batch_targets

    def _get_batch_internal(self, batch_size: int) -> tuple[Int[Tensor, "bs context"], Int[Tensor, "bs context"]]:
        """
        Get a batch of data from memory mapped x: np.ndarray

        If run in distributed context, will return correct chunk of data instead of full batch size
        """
        if self.is_distributed:
            # We want to split unevenly across workers
            target_log_ranks = [0]
            small_batch_size = batch_size // self.world_size

            if (
                self.world_size == 4
                and any(["RTX 5090" in s for s in self.gpu_names])
                and any(["RTX 4090" in s for s in self.gpu_names])
            ):
                target_log_ranks = [i for i in range(self.world_size) if "RTX 5090" in self.gpu_names[i]]
                adj_n_chunks = 10
                small_batch_size = (batch_size // adj_n_chunks) * 2
                large_batch_size = (batch_size // adj_n_chunks) * 3
                assert small_batch_size * 2 + large_batch_size * 2 == batch_size, "batch sizes don't match"
                local_batch_size = large_batch_size if self.rank in target_log_ranks else small_batch_size
            else:
                local_batch_size = small_batch_size
        else:
            local_batch_size = batch_size
        self.local_batch_size = local_batch_size

        # sampler (will be sampled deterministically at all devices)
        sampled_indices = torch.randint(low=0, high=len(self), size=(batch_size,), generator=self.g)

        if self.is_distributed:
            # we split the work unevenly
            if self.rank in target_log_ranks:
                offset = self.rank * small_batch_size
            else:
                offset = self.rank * local_batch_size
            local_sampled_indices = sampled_indices[offset : offset + local_batch_size]
        else:
            local_sampled_indices = sampled_indices

        batch_inputs, batch_targets = self._gather_batch_numpy(local_sampled_indices.cpu().numpy())
        batch_inputs, batch_targets = torch.from_numpy(batch_inputs), torch.from_numpy(batch_targets)
        return batch_inputs, batch_targets

    def _prefetch(self, batch_size):
        batch_inputs, batch_targets = self._get_batch_internal(batch_size)

        if torch.cuda.is_available():
            batch_inputs = batch_inputs.pin_memory()
            batch_targets = batch_targets.pin_memory()

        with self._prefetch_lock:
            self._prefetch_batch = (batch_inputs, batch_targets)
            self._prefetch_event.set()  # mark ready

    def get_batch(self, batch_size):
        # First ever call: compute synchronously
        if self._prefetch_batch is None and not self._prefetch_event.is_set():
            self._prefetch(batch_size)

        # Wait until a prefetched batch is ready
        self._prefetch_event.wait()

        # Take current batch
        with self._prefetch_lock:
            out = self._prefetch_batch
            # reset the flag so the next prefetch can signal again
            self._prefetch_event.clear()

        # Kick off prefetch of the *next* batch in the background
        threading.Thread(target=self._prefetch, args=(batch_size,), daemon=True).start()

        return out


# ds of len 10
# context 5
# we need to sample from [0, 5) 0, 1, 2, 3, 4; 4:10
