import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda import Event
from prettytable import PrettyTable

BACKEND = "nccl"


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if BACKEND != "gloo":
        torch.cuda.set_device(f"cuda:{rank}")
    dist.init_process_group(BACKEND, rank=rank, world_size=world_size)


def shutdown():
    dist.destroy_process_group()


def distributed_demo(rank, world_size):
    setup(rank, world_size)
    device = "cpu" if BACKEND == "gloo" else "cuda"
    is_distributed = dist.is_initialized()
    print(f"{is_distributed=}")

    table = PrettyTable(["MB", "MB / s"])
    table.title = f"{BACKEND=} {world_size=}"
    size_of_mb = 262_144
    start, end = Event(enable_timing=True), Event(enable_timing=True)
    times = [None for _ in range(world_size)]
    for mb in [1, 10, 100, 1000]:
        n_items = size_of_mb * mb
        data = torch.randn(n_items, device=device, dtype=torch.float32)
        # print(f"rank {rank} data (before all-reduce): {data}")
        for _ in range(5):
            dist.all_reduce(data, async_op=False)

        torch.cuda.synchronize()
        rep = 10
        start.record()
        for _ in range(rep):
            dist.all_reduce(data, async_op=False)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        dist.all_gather_object(times, ms)
        mean_ms = np.mean(times)
        table.add_row([mb, mb / (mean_ms / 1e3)])
        # print(f"rank {rank} data (after all-reduce): {data}")
    if rank == 0:
        print(table)


    table = PrettyTable(["MB", "MB / s"])
    table.title = f"{BACKEND=} {world_size=} 5 buckets"
    times = [None for _ in range(world_size)]
    for mb in [1, 10, 100, 1000]:
        n_items = size_of_mb * mb
        n_buckets = 5
        data = torch.randn(n_items // n_buckets, device=device)
        # print(f"rank {rank} data (before all-reduce): {data}")
        for _ in range(5):
            for _ in range(n_buckets):
                dist.all_reduce(data, async_op=False)

        torch.cuda.synchronize()
        rep = 10
        start.record()
        for _ in range(rep):
            for _ in range(n_buckets):
                dist.all_reduce(data, async_op=False)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        dist.all_gather_object(times, ms)
        mean_ms = np.mean(times)
        table.add_row([mb, mb / (mean_ms / 1e3)])
        # print(f"rank {rank} data (after all-reduce): {data}")
    if rank == 0:
        print(table)

    table = PrettyTable(["MB", "MB / s"])
    table.title = f"{BACKEND=} {world_size=} 50 buckets"
    times = [None for _ in range(world_size)]
    for mb in [1, 10, 100, 1000]:
        n_items = size_of_mb * mb
        n_buckets = 50
        data = torch.randn(n_items // n_buckets, device=device)
        # print(f"rank {rank} data (before all-reduce): {data}")
        for _ in range(5):
            for _ in range(n_buckets):
                dist.all_reduce(data, async_op=False)

        torch.cuda.synchronize()
        rep = 10
        start.record()
        for _ in range(rep):
            for _ in range(n_buckets):
                dist.all_reduce(data, async_op=False)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        dist.all_gather_object(times, ms)
        mean_ms = np.mean(times)
        table.add_row([mb, mb / (mean_ms / 1e3)])
        # print(f"rank {rank} data (after all-reduce): {data}")
    if rank == 0:
        print(table)
    shutdown()


if __name__ == "__main__":
    world_size = 3
    mp.spawn(fn=distributed_demo, args=(world_size,), nprocs=world_size, join=True)
