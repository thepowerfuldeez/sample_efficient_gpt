import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from sample_efficient_gpt.utils.profiling import nvtx_range


class DDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        if dist.is_initialized():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)
        for p in self.module.parameters():
            if p.is_leaf and p.requires_grad:
                p.register_post_accumulate_grad_hook(self._hook)
        self._should_all_reduce = True
        self.handles = []

    def _hook(self, p: Tensor) -> None:
        if p.grad is not None and self._should_all_reduce:
            with nvtx_range("all-reduce hook"):
                # TODO: compress grads to bf16 and do sum accumulation in fp32
                handle = dist.all_reduce(p.grad, dist.ReduceOp.SUM, async_op=True)
                self.handles.append(handle)

    @contextmanager
    def no_sync(self):
        before = self._should_all_reduce
        self._should_all_reduce = False
        try:
            yield
        finally:
            self._should_all_reduce = before

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> float:
        torch.cuda.synchronize()
        t0 = time.monotonic()

        for h in self.handles:
            h.wait()
        self.handles.clear()

        # for p in self.module.parameters():
        #     if p.grad is not None:
        #         dist.all_reduce(p.grad)

        torch.cuda.synchronize()
        time_comm = time.monotonic() - t0
        return time_comm
