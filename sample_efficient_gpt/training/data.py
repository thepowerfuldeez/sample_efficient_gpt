from pathlib import Path
from collections.abc import Iterator

import torch
import numpy as np
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
    ):
        """
        Reads numpy file in memory mapped mode
        Samples `context_length` chunks
        """
        if isinstance(path_or_ds, np.ndarray):
            self.ds = path_or_ds
        else:
            self.ds = np.load(path_or_ds, mmap_mode="r")
            # npz is a dict-like with key 'values'
            if hasattr(self.ds, "files"):
                self.ds = self.ds["values"]
        self.total_length = self.ds.shape[0]
        logger.info(f"Dataset length: {self.total_length}")
        self.context_length = context_length
        self.device = device
        if seed is not None:
            self.g = torch.Generator().manual_seed(seed)
        else:
            self.g = None

    def __getitem__(self, i: int) -> tuple[Int[Tensor, "context"], Int[Tensor, "context"]]:
        """
        i can be anything from 0 to self.__len__
        """
        i_finish = i + self.context_length + 1
        # print(len(self.ds), self.__len__(), i, i_finish)
        chunk = self.ds[i:i_finish].astype(np.int32)
        inputs = torch.from_numpy(chunk[:-1]).to(device=self.device)
        targets = torch.from_numpy(chunk[1:]).to(device=self.device)
        return inputs, targets

    def __len__(self):
        return self.total_length - self.context_length

    def get_iterator(self, batch_size: int) -> Iterator[tuple[Int[Tensor, "bs context"], Int[Tensor, "bs context"]]]:
        """
        Return an iterator of batches in a sequential order
        Last batch is dropped
        """
        sequential_indices = torch.arange(start=0, end=len(self), step=batch_size * self.context_length)
        for i_start in sequential_indices:
            batch_inputs = torch.empty((batch_size, self.context_length), device=self.device, dtype=torch.int32)
            batch_targets = torch.empty((batch_size, self.context_length), device=self.device, dtype=torch.int32)
            for sample_idx in range(batch_size):
                inputs, targets = self.__getitem__(i_start.item() + sample_idx)
                batch_inputs[sample_idx] = inputs
                batch_targets[sample_idx] = targets
            yield batch_inputs, batch_targets

    def get_batch(self, batch_size: int) -> tuple[Int[Tensor, "bs context"], Int[Tensor, "bs context"]]:
        """
        Get a batch of data from memory mapped x: np.ndarray
        """
        batch_inputs = torch.empty((batch_size, self.context_length), device=self.device, dtype=torch.int32)
        batch_targets = torch.empty((batch_size, self.context_length), device=self.device, dtype=torch.int32)
        # sampler
        sampled_indices = torch.randint(low=0, high=len(self), size=(batch_size,), generator=self.g)
        for sample_idx in range(batch_size):
            inputs, targets = self.__getitem__(sampled_indices[sample_idx].item())
            batch_inputs[sample_idx] = inputs
            batch_targets[sample_idx] = targets
        return batch_inputs, batch_targets


# ds of len 10
# context 5
# we need to sample from [0, 5) 0, 1, 2, 3, 4; 4:10
