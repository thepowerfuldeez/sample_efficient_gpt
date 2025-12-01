import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import time
from collections.abc import Callable

from sample_efficient_gpt.transformer.core import Linear
from sample_efficient_gpt.train import apply_overrides, cfg, Trainer
from sample_efficient_gpt.training.distributed import DDP
from sample_efficient_gpt.training.fp8_utils import convert_to_float8_training


# torch.cuda.set_device("cuda:2")

class Lin(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = Linear(2048, 4096)
        self.l2 = Linear(4096, 4096)
        self.l3 = Linear(4096, 4096)
        self.l4 = Linear(4096, 4096)
        self.l5 = Linear(4096, 4096)
        self.l6 = Linear(4096, 128)
        self.l7 = Linear(128, 1)
    def forward(self, x):
        for l in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7]:
            x = l(x)
        return x

if __name__ == "__main__":
    dist.init_process_group("nccl")
    r = dist.get_rank()
    # create model and sample input
    # m = (
    #     nn.Sequential(
    #         Linear(2048, 4096),
    #         Linear(4096, 4096),
    #         Linear(4096, 4096),
    #         Linear(4096, 4096),
    #         Linear(4096, 4096),
    #         Linear(4096, 128),
    #         Linear(128, 1),
    #     )
    #     .bfloat16()
    #     .cuda()
    # )
    device = f"cuda:{r}"
    m = Lin().bfloat16()
    m = m.to(device)
    m.train()
    x = torch.randn(4096, 2048, device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    # convert specified `torch.nn.Linear` modules to `Float8Linear`


    # enable torch.compile for competitive performance
    convert_to_float8_training(m)
    print(m)
    m = torch.compile(m)
    m = DDP(m)
    r = dist.get_rank()

    # warmup
    for _ in range(5):
        optimizer.zero_grad()
        output = m(x)
        # use fake labels for demonstration purposes
        fake_labels = torch.ones_like(output)
        loss = F.mse_loss(output, fake_labels)
        loss.backward()
        m.finish_gradient_synchronization()
        optimizer.step()

    # toy training loop
    for _ in range(10):
        optimizer.zero_grad()
        output = m(x)
        # use fake labels for demonstration purposes
        fake_labels = torch.ones_like(output)
        loss = F.mse_loss(output, fake_labels)
        loss.backward()
        m.finish_gradient_synchronization()
        optimizer.step()

    t0 = time.monotonic()
    # toy training loop
    s = 0
    for _ in range(100):
        optimizer.zero_grad()
        output = m(x)
        # use fake labels for demonstration purposes
        fake_labels = torch.ones_like(output)
        loss = F.mse_loss(output, fake_labels)
        loss.backward()
        s = time.monotonic() - t0
        m.finish_gradient_synchronization()
        optimizer.step()
    print(r, s)

    print(time.monotonic() - t0)
