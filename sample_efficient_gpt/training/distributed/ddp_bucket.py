import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor


class DDP(nn.Module):
    def __init__(self, module, bucket_size_mb: float = 9.7):
        super().__init__()
        self.module = module
        self.bucket_size = int(bucket_size_mb * 1024 * 1024)

        # p name -> bucket_idx
        self.buckets_map = {}
        assert dist.is_initialized()
        cur_bucket_id = 0
        cur_bucket_size = 0
        prev_dt = None

        for n, p in list(self.module.named_parameters())[::-1]:
            dist.broadcast(p.data, src=0)
            if not (p.is_leaf and p.requires_grad):
                continue
            tensor_bytes = p.data.numel() * p.data.element_size()

            dt = p.data.dtype
            # start new bucket if dtype changes or size would overflow
            if (prev_dt is not None and dt != prev_dt) or cur_bucket_size + tensor_bytes > self.bucket_size:
                cur_bucket_id += 1
                cur_bucket_size = 0

            cur_bucket_size += tensor_bytes
            self.buckets_map[n] = cur_bucket_id
            prev_dt = dt
        assert len(self.buckets_map)

        # bucket ix -> param names
        self.bucket_to_names = {}
        for n, bucket_idx in self.buckets_map.items():
            self.bucket_to_names.setdefault(bucket_idx, []).append(n)

        for n, p in self.module.named_parameters():
            if p.is_leaf and p.requires_grad:
                bucket_idx = self.buckets_map[n]
                param_name = n

                def make_hook(bucket_idx=bucket_idx, param_name=param_name):
                    # torch hook expect only param as an input
                    def _hook_inner(p: Tensor) -> None:
                        return self._hook(bucket_idx, param_name, p)

                    return _hook_inner

                p.register_post_accumulate_grad_hook(make_hook())
        self._should_all_reduce = True
        self.handles = []

        self.total_bytes = 0
        # bucket idx -> group, initialize buckets with None
        self.buckets = {
            bucket_idx: [None for _ in range(len(names))] for bucket_idx, names in self.bucket_to_names.items()
        }
        self.comm_stream = torch.cuda.Stream()

    def _hook(self, bucket_idx: int, param_name: str, p: Tensor) -> None:
        """
        Main backward hook with future that would unflatten the param group
        We would construct param group deterministicaly by name until order matches 1-1
        """
        if p.grad is not None and self._should_all_reduce:
            g = p.grad
            tensor_bytes = g.numel() * g.element_size()
            self.total_bytes += tensor_bytes

            # self.buckets.setdefault(bucket_idx, []).append(g)
            # find position
            # print(f"[rank={dist.get_rank()}] looking for {param_name} in {self.bucket_to_names[bucket_idx]} with {bucket_idx=}")
            grad_position = self.bucket_to_names[bucket_idx].index(param_name)
            self.buckets[bucket_idx][grad_position] = g
            # print(
            #     f"[rank={dist.get_rank()}] grad shape {g.shape}, collecting grad to a bucket {bucket_idx}, cur count {len(self.buckets[bucket_idx])}, needed {len(self.bucket_to_names[bucket_idx])}"
            # )

            # no more None left
            bucket_is_full = not [x for x in self.buckets[bucket_idx] if x is None]
            if bucket_is_full:
                group = self.buckets.pop(bucket_idx)
                self.buckets[bucket_idx] = [None for _ in range(len(group))]
                flat = torch._utils._flatten_dense_tensors(group)
                ready = torch.cuda.Event()
                torch.cuda.current_stream().record_event(ready)
                with torch.cuda.stream(self.comm_stream):
                    self.comm_stream.wait_event(ready)
                    handle = dist.all_reduce(flat.bfloat16(), dist.ReduceOp.SUM, async_op=True)

                    def _on_finish(_, flat=flat, group=group):
                        flat = flat.to(group[0].dtype)
                        torch._utils._unflatten_dense_tensors(flat, group)
                handle.get_future().then(_on_finish)
                # print(
                #     f"[rank={dist.get_rank()}] grad shape {g.shape} flat shape {flat.shape}, sending all reduce with bucket {bucket_idx}, "
                #     f"now bucket_idx is in self.buckets: {bucket_idx in self.buckets}"
                # )
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
        t0 = time.monotonic()

        for h in self.handles:
            h.wait()
        self.handles.clear()

        time_comm = time.monotonic() - t0
        # speed = (self.total_bytes / (1024 * 1024)) / time_comm
        # print(f"total bytes: {self.total_bytes}, communication speed {speed} GB / s")
        self.total_bytes = 0
        return time_comm
