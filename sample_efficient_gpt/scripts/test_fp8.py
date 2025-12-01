import torch
from torch import nn
import torch.nn.functional as F
import time
from collections.abc import Callable

from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.config import Float8LinearConfig, Float8LinearRecipeName, Float8GemmConfig
from torchao.float8.float8_linear import (
    LinearMMConfig,
    ScaledMMConfig,
    matmul_with_hp_or_float8_args,
    ScalingType,
    WeightWithDynamicFloat8CastTensor,
)
from torchao.float8 import convert_to_float8_training

from sample_efficient_gpt.transformer.core import Linear
from sample_efficient_gpt.train import apply_overrides, cfg, Trainer


torch.cuda.set_device("cuda:2")

if __name__ == "__main__":

    # create model and sample input
    m = (
        nn.Sequential(
            Linear(2048, 4096),
            Linear(4096, 4096),
            Linear(4096, 4096),
            Linear(4096, 4096),
            Linear(4096, 4096),
            Linear(4096, 128),
            Linear(128, 1),
        )
        .bfloat16()
        .cuda()
    )
    x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    # optional: filter modules from being eligible for float8 conversion
    def module_filter_fn(mod: torch.nn.Module, fqn: str):
        # don't convert the last module
        # if fqn == "1":
        #     return False
        print(fqn)
        # don't convert linear modules with weight dimensions not divisible by 16
        if isinstance(mod, nn.Linear):
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
        return True

    # convert specified `torch.nn.Linear` modules to `Float8Linear`
    convert_to_float8_training(m, module_filter_fn=module_filter_fn)

    print(m)

    # enable torch.compile for competitive performance
    m = torch.compile(m)

    # warmup
    for _ in range(5):
        optimizer.zero_grad()
        output = m(x)
        # use fake labels for demonstration purposes
        fake_labels = torch.ones_like(output)
        loss = F.mse_loss(output, fake_labels)
        loss.backward()
        optimizer.step()

    # toy training loop
    for _ in range(10):
        optimizer.zero_grad()
        output = m(x)
        # use fake labels for demonstration purposes
        fake_labels = torch.ones_like(output)
        loss = F.mse_loss(output, fake_labels)
        loss.backward()
        optimizer.step()

    t0 = time.monotonic()
    # toy training loop
    for _ in range(100):
        optimizer.zero_grad()
        output = m(x)
        # use fake labels for demonstration purposes
        fake_labels = torch.ones_like(output)
        loss = F.mse_loss(output, fake_labels)
        loss.backward()
        optimizer.step()

    print(time.monotonic() - t0)
