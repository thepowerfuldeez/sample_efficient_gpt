import math
from typing import Any
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-4):
        super().__init__(params=params, defaults=dict(lr=lr))

    def step(self, closure: Any | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 1e-6,
        eps: float = 1e-8,
    ):
        super().__init__(params=params, defaults=dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps))

    # @nvtx.range("adamw step")
    def step(self, closure: Any | None = None):
        loss = None if closure is None else closure()
        total_update_sq, total_weight_sq = 0.0, 0.0
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                # init first and second order moments if we haven't already
                m: Tensor = state.get("m", torch.zeros_like(p.data, requires_grad=False))
                v: Tensor = state.get("v", torch.zeros_like(p.data, requires_grad=False))

                grad: Tensor = p.grad.data

                m: Tensor = beta1 * m + (1 - beta1) * grad
                v: Tensor = beta2 * v + (1 - beta2) * torch.square(grad)
                bias_correction_term: float = math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # ! wd is decoupled, it should go first !
                wd_delta = -lr * wd * p.data
                adam_delta = -lr * bias_correction_term * m / (torch.sqrt(v) + eps)
                delta = wd_delta + adam_delta

                total_update_sq += (delta.float().norm() ** 2).item()
                total_weight_sq += (p.data.float().norm() ** 2).item()

                p.data.add_(delta)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        update_ratio = math.sqrt(total_update_sq) / (math.sqrt(total_weight_sq) + eps)
        return (loss, update_ratio)


def get_cosine_lr(t: int, lr_max: float, lr_min: float, warmup_steps: int, cosine_steps: int) -> float:
    """
    Update learning rate based on cosine schedule with warmup

    t: int - current step
    lr_max: float - max learning rate (usually set as original learning rate)
    lr_min: float - minimum learning rate after decay
    warmup_steps: int - warmup steps starting from ~0 (t / warmup_steps) * lr_max
    cosine_steps: int - total number of steps in the cosine schedule, starting from warmup_steps to cosine steps

    Returns: updated lr
    """
    if t < warmup_steps:
        return t / warmup_steps * lr_max
    elif warmup_steps <= t < cosine_steps:
        cos_lr: float = lr_min + 0.5 * (1 + math.cos((t - warmup_steps) / (cosine_steps - warmup_steps) * math.pi)) * (
            lr_max - lr_min
        )
        return cos_lr
    else:
        return lr_min


def get_wsd_lr(t: int, lr_max: float, lr_min: float, warmup_steps: int, stable_steps: int, decay_steps: int) -> float:
    """
    Update learning rate based on Warmup Stable Decay schedule

    t: int - current step
    lr_max: float - max learning rate (usually set as original learning rate)
    lr_min: float - minimum learning rate after decay
    warmup_steps: int - warmup steps starting from ~0 (t / warmup_steps) * lr_max
    stable_steps: int - total number of steps in the stable state [warmup_steps, decay_steps]
    decay_steps: int - total number of steps in the decay state to lr_min [decay_steps, total_steps]

    Returns: updated lr
    """
    if t < warmup_steps:
        return t / warmup_steps * lr_max
    elif warmup_steps <= t < warmup_steps + stable_steps:
        return lr_max
    else:
        # t >= warmup_steps + stable_steps
        return (1 - (t - warmup_steps - stable_steps) / decay_steps) * (lr_max - lr_min)


# @nvtx.range("clip grad")
def clip_grad_norm_(params: Iterable[nn.Parameter], max_grad_norm: float = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Clips gradients to `max_grad_norm` in-place
    """
    grads = [p.grad for p in params if p.grad is not None]
    assert len(grads), "grads are empty!"
    total_squared_norm = torch.zeros((1,), dtype=grads[0].dtype, device=grads[0].device)
    for g in grads:
        total_squared_norm += torch.linalg.norm(g) ** 2
    norm = total_squared_norm.sqrt()
    if norm >= max_grad_norm:
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(max_grad_norm / (norm + eps))
    return norm


if __name__ == "__main__":
    weights = nn.Parameter(torch.randn(10, 10))
    opt = SGD([weights], lr=1e0)

    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).norm()
        print(loss.item())
        loss.backward()
        opt.step()
