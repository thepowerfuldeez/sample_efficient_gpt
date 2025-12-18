import math

import torch
import torch.nn as nn
from torch import Tensor

from sample_efficient_gpt.transformer.core import Linear, SwiGLU


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class TopKMoE(nn.Module):
    """
    Simple token-level MoE (no expert-parallelism) for quick experimentation.

    - Routes each token to top-k experts.
    - Computes a Switch-style load balancing auxiliary loss.
    - Optional capacity factor drops overflow tokens (keeps first tokens per expert after sorting).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 1,
        capacity_factor: float = 1.0,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.0,
        router_jitter: float = 0.0,
        normalize_gates: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if capacity_factor <= 0:
            raise ValueError("capacity_factor must be > 0")

        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.capacity_factor = float(capacity_factor)
        self.aux_loss_coef = float(aux_loss_coef)
        self.z_loss_coef = float(z_loss_coef)
        self.router_jitter = float(router_jitter)
        self.normalize_gates = bool(normalize_gates)

        self.router = Linear(d_model, self.num_experts, device=device, dtype=dtype)
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, d_ff, device=device, dtype=dtype) for _ in range(self.num_experts)])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: [B, S, D]
        b, s, d = x.shape
        tokens = b * s
        x_flat = x.reshape(tokens, d)

        router_logits = self.router(x_flat)  # [T, E]
        if self.router_jitter:
            router_logits = router_logits + self.router_jitter * torch.randn_like(router_logits)

        # Softmax in fp32 for stability, keep probs in fp32 (used only for routing and aux loss).
        router_probs = torch.softmax(router_logits.float(), dim=-1)  # [T, E]

        topk_probs, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1)  # [T, K]
        if self.normalize_gates and self.top_k > 1:
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Switch-style load balancing loss uses primary expert assignments.
        primary_idx = topk_idx[:, 0]
        load = torch.bincount(primary_idx, minlength=self.num_experts).float() / float(tokens)  # [E]
        importance = router_probs.sum(dim=0) / float(tokens)  # [E]
        load_balance_loss = (importance * load).sum() * float(self.num_experts)

        z_loss = router_logits.float().logsumexp(dim=-1).pow(2).mean()
        aux_loss = self.aux_loss_coef * load_balance_loss + self.z_loss_coef * z_loss

        # Dispatch: treat top-k by duplicating tokens K times, then reduce back.
        if self.top_k == 1:
            x_rep = x_flat
            expert_idx = topk_idx[:, 0]
            gate = topk_probs[:, 0]
            tokens_per_expert = tokens
        else:
            x_rep = x_flat.repeat_interleave(self.top_k, dim=0)  # [T*K, D]
            expert_idx = topk_idx.reshape(-1)  # [T*K]
            gate = topk_probs.reshape(-1)  # [T*K]
            tokens_per_expert = tokens * self.top_k

        # Capacity per expert (as in Switch: proportional to average tokens/expert).
        capacity = int(math.ceil(self.capacity_factor * (tokens_per_expert / self.num_experts)))
        capacity = max(1, capacity)

        order = torch.argsort(expert_idx)
        x_sorted = x_rep[order]
        gate_sorted = gate[order].to(dtype=x.dtype)
        expert_sorted = expert_idx[order]

        out_sorted = torch.zeros_like(x_sorted)
        counts = torch.bincount(expert_sorted, minlength=self.num_experts)
        offsets = torch.cumsum(counts, dim=0)

        start = 0
        for expert_id in range(self.num_experts):
            end = int(offsets[expert_id].item())
            n = end - start
            if n <= 0:
                start = end
                continue

            take = min(n, capacity)
            if take > 0:
                x_e = x_sorted[start : start + take]
                y_e = self.experts[expert_id](x_e)
                y_e = y_e * gate_sorted[start : start + take].unsqueeze(-1)
                out_sorted[start : start + take] = y_e
            start = end

        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        out_rep = out_sorted[inv]

        if self.top_k == 1:
            out = out_rep
        else:
            out = out_rep.reshape(tokens, self.top_k, d).sum(dim=1)

        return out.reshape(b, s, d), aux_loss.to(dtype=x.dtype)

