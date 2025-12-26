import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from sample_efficient_gpt.transformer.core import SwiGLU


class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class TopKMoE(nn.Module):
    """
    Token-choice top-K MoE (single-process, no expert-parallel).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 1,
        capacity_factor: float = 1.0,
        aux_loss_coef: float = 0.01,
        router_jitter: float = 0.0,
        normalize_gates: bool = True,
        gate_scale: float = 1.0,
        num_shared_experts: int = 0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if capacity_factor < 0:
            raise ValueError("capacity_factor must be >= 0 (0 disables capacity drops)")
        if num_shared_experts < 0 or num_shared_experts >= num_experts:
            raise ValueError("num_shared_experts must be in [0, num_experts)")

        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.num_experts = int(num_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.num_routed_experts = int(num_experts - num_shared_experts)
        if self.num_routed_experts <= 0:
            raise ValueError("num_routed_experts must be > 0")
        if top_k <= 0 or top_k > self.num_routed_experts:
            raise ValueError("top_k must be in [1, num_routed_experts]")

        self.top_k = int(top_k)
        self.capacity_factor = float(capacity_factor)
        self.aux_loss_coef = float(aux_loss_coef)
        self.router_jitter = float(router_jitter)
        self.normalize_gates = bool(normalize_gates)
        self.gate_scale = float(gate_scale)

        self.router = nn.Linear(self.d_model, self.num_routed_experts, bias=True, device=device, dtype=dtype)
        self.experts = nn.ModuleList(
            [ExpertMLP(self.d_model, self.d_ff, device=device, dtype=dtype) for _ in range(self.num_experts)]
        )
        self.last_stats: Optional[Dict[str, Tensor]] = None

    @torch.compile(disable=True)
    def _aux_loss(self, router_probs: Tensor, expert_idx_flat: Tensor, denom: int) -> Tensor:
        if self.aux_loss_coef == 0.0:
            return torch.zeros((), device=router_probs.device, dtype=router_probs.dtype)

        importance = router_probs.mean(dim=0)  # [E], grad flows
        load_counts = torch.bincount(expert_idx_flat.to(torch.int64), minlength=self.num_routed_experts).to(torch.float32)
        denom_f = float(max(int(denom), 1))
        load = load_counts / denom_f
        loss = (importance * load).sum() * float(self.num_routed_experts)
        return (self.aux_loss_coef * loss).to(dtype=router_probs.dtype)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        b, s, d = x.shape
        tokens = b * s
        x_flat = x.reshape(tokens, d)

        router_logits = self.router(x_flat)  # [T, E_routed]
        if self.router_jitter:
            router_logits = router_logits + self.router_jitter * torch.randn_like(router_logits)
        router_probs = torch.softmax(router_logits.float(), dim=-1)  # [T, E_routed] float32

        shared_out = None
        if self.num_shared_experts > 0:
            shared_out = torch.zeros_like(x_flat)
            for e in range(self.num_shared_experts):
                shared_out = shared_out + self.experts[e](x_flat)
            if self.num_shared_experts > 1:
                shared_out = shared_out / float(self.num_shared_experts)

        if self.top_k == 1:
            gate, expert_idx = router_probs.max(dim=-1)  # [T], [T]
            if self.gate_scale != 1.0:
                gate = gate * self.gate_scale
            aux_loss = self._aux_loss(router_probs, expert_idx, denom=tokens)

            if self.capacity_factor == 0.0:
                capacity = tokens
            else:
                capacity = int(math.ceil(self.capacity_factor * (tokens / self.num_routed_experts)))
                capacity = max(1, capacity)

            order = torch.argsort(expert_idx)
            x_sorted = x_flat[order]
            gate_sorted = gate[order].to(dtype=x.dtype)
            expert_sorted = expert_idx[order]

            out_sorted = torch.zeros_like(x_sorted)
            counts = torch.bincount(expert_sorted, minlength=self.num_routed_experts)
            offsets = torch.cumsum(counts, dim=0)

            start = 0
            for e in range(self.num_routed_experts):
                end = int(offsets[e].item())
                n = end - start
                if n <= 0:
                    start = end
                    continue
                take = min(n, capacity)
                if take > 0:
                    expert = self.experts[e + self.num_shared_experts]
                    y = expert(x_sorted[start : start + take])
                    y = y * gate_sorted[start : start + take].unsqueeze(-1)
                    out_sorted[start : start + take] = y
                start = end

            inv = torch.empty_like(order)
            inv[order] = torch.arange(order.numel(), device=order.device)
            out = out_sorted[inv].reshape(b, s, d)
            if shared_out is not None:
                out = out + shared_out.reshape(b, s, d)

            self._maybe_log_stats(expert_idx, gate, router_logits)
            return out, aux_loss.to(dtype=x.dtype)

        topk_probs, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1)  # [T, K]
        if self.normalize_gates:
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        gate = topk_probs.reshape(-1)  # [T*K]
        expert_idx = topk_idx.reshape(-1)  # [T*K]
        if self.gate_scale != 1.0:
            gate = gate * self.gate_scale

        aux_loss = self._aux_loss(router_probs, expert_idx, denom=tokens * self.top_k)

        assignments = tokens * self.top_k
        if self.capacity_factor == 0.0:
            capacity = assignments
        else:
            capacity = int(math.ceil(self.capacity_factor * (assignments / self.num_routed_experts)))
            capacity = max(1, capacity)

        x_rep = x_flat.repeat_interleave(self.top_k, dim=0)  # [T*K, D]

        order = torch.argsort(expert_idx)
        x_sorted = x_rep[order]
        gate_sorted = gate[order].to(dtype=x.dtype)
        expert_sorted = expert_idx[order]

        out_sorted = torch.zeros_like(x_sorted)
        counts = torch.bincount(expert_sorted, minlength=self.num_routed_experts)
        offsets = torch.cumsum(counts, dim=0)

        start = 0
        for e in range(self.num_routed_experts):
            end = int(offsets[e].item())
            n = end - start
            if n <= 0:
                start = end
                continue
            take = min(n, capacity)
            if take > 0:
                expert = self.experts[e + self.num_shared_experts]
                y = expert(x_sorted[start : start + take])
                y = y * gate_sorted[start : start + take].unsqueeze(-1)
                out_sorted[start : start + take] = y
            start = end

        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        out_rep = out_sorted[inv]  # [T*K, D]
        out = out_rep.reshape(tokens, self.top_k, d).sum(dim=1).reshape(b, s, d)
        if shared_out is not None:
            out = out + shared_out.reshape(b, s, d)

        self._maybe_log_stats(expert_idx, gate, router_logits)
        return out, aux_loss.to(dtype=x.dtype)

    def _maybe_log_stats(self, expert_idx_flat: Tensor, gate_flat: Tensor, router_logits: Tensor) -> None:
        if os.environ.get("SEGPT_LOG_MOE_STATS", "0") != "1":
            return
        with torch.no_grad():
            counts = torch.bincount(expert_idx_flat.to(torch.int64), minlength=self.num_routed_experts).to(torch.float32)
            denom = counts.sum().clamp_min(1.0)
            p = counts / denom
            ent = -(p * (p.clamp_min(1e-20).log())).sum()
            ent_norm = ent / float(math.log(self.num_routed_experts)) if self.num_routed_experts > 1 else ent
            self.last_stats = {
                "load_max_frac": p.max(),
                "load_ent_norm": ent_norm,
                "gate_mean": gate_flat.mean().to(torch.float32),
                "logits_std": router_logits.float().std(),
            }
