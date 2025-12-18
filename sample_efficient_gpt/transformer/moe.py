import math

import torch
import torch.nn as nn
import torch.distributed as dist
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
    Token-level MoE with an optional expert-parallel path (EP==world_size).

    - EP disabled: all experts are replicated per rank (fast to prototype, expensive at scale).
    - EP enabled: experts are sharded across ranks and tokens are dispatched via all-to-all.

    Notes:
    - EP currently assumes `expert_parallel_size == world_size` (single replica).
    - EP currently supports `top_k == 1` (Switch-style top-1 routing).
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
        expert_parallel_size: int = 1,
        expert_precision: str = "bf16",
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
        if expert_parallel_size <= 0:
            raise ValueError("expert_parallel_size must be > 0")

        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.capacity_factor = float(capacity_factor)
        self.aux_loss_coef = float(aux_loss_coef)
        self.z_loss_coef = float(z_loss_coef)
        self.router_jitter = float(router_jitter)
        self.normalize_gates = bool(normalize_gates)
        self.expert_parallel_size = int(expert_parallel_size)
        self.expert_precision = str(expert_precision)
        self._fp8_converted = False

        # Bias matters for stable initialization / upcycling.
        self.router = nn.Linear(self.d_model, self.num_experts, bias=True, device=device, dtype=dtype)

        self._ep_enabled = False
        self._ep_world_size = 1
        self._ep_rank = 0
        self._num_local_experts = self.num_experts
        if dist.is_available() and dist.is_initialized() and self.expert_parallel_size > 1:
            self._ep_world_size = dist.get_world_size()
            self._ep_rank = dist.get_rank()
            if self.expert_parallel_size != self._ep_world_size:
                raise ValueError(
                    f"Only expert_parallel_size == world_size is supported for now: "
                    f"{self.expert_parallel_size} != {self._ep_world_size}"
                )
            if self.top_k != 1:
                raise ValueError("EP MoE currently supports top_k == 1 only.")
            if self.num_experts % self._ep_world_size != 0:
                raise ValueError(
                    f"num_experts ({self.num_experts}) must be divisible by world_size ({self._ep_world_size})"
                )
            self._ep_enabled = True
            self._num_local_experts = self.num_experts // self._ep_world_size

        if self._ep_enabled:
            self.experts = nn.ModuleList(
                [ExpertMLP(self.d_model, self.d_ff, device=device, dtype=dtype) for _ in range(self._num_local_experts)]
            )
            for p in self.experts.parameters():
                setattr(p, "_expert_parallel", True)
        else:
            self.experts = nn.ModuleList(
                [ExpertMLP(self.d_model, self.d_ff, device=device, dtype=dtype) for _ in range(self.num_experts)]
            )

    def convert_experts_to_fp8(self) -> None:
        """
        Best-effort conversion of *expert* matmuls to float8 training (torchao).

        Important:
        - Call this once after moving the model to CUDA and before building the optimizer / DDP buckets.
        - This converts only the experts (not the router / attention / dense path).
        """
        if self.expert_precision.lower() != "fp8" or self._fp8_converted:
            return
        # Experts must be on CUDA for float8 kernels.
        any_cuda = any(p.is_cuda for p in self.experts.parameters())
        if not any_cuda:
            return
        try:
            from torchao.float8 import (
                CastConfig,
                Float8LinearConfig,
                Float8GemmConfig,
                ScalingType,
                convert_to_float8_training,
            )
        except Exception as e:
            raise RuntimeError(
                "moe_expert_precision='fp8' requires torchao.float8 to be installed and importable."
            ) from e

        config = Float8LinearConfig(
            pad_inner_dim=True,
            cast_config_input=CastConfig(scaling_type=ScalingType.DYNAMIC),
            cast_config_weight=CastConfig(scaling_type=ScalingType.DYNAMIC),
            cast_config_grad_output=CastConfig(scaling_type=ScalingType.DYNAMIC),
            gemm_config_grad_input=Float8GemmConfig(use_fast_accum=True),
            gemm_config_grad_weight=Float8GemmConfig(use_fast_accum=True),
            gemm_config_output=Float8GemmConfig(use_fast_accum=True),
        )

        # Convert only expert MLP matmuls; keep router/attention/dense in higher precision.
        for expert in self.experts:
            convert_to_float8_training(expert, config=config)

        # Conversion may replace submodules/parameters; re-apply EP marking.
        if self._ep_enabled:
            for p in self.experts.parameters():
                setattr(p, "_expert_parallel", True)
        self._fp8_converted = True

    def _aux_loss(self, router_logits: Tensor, router_probs: Tensor, top1_idx: Tensor) -> Tensor:
        if self.aux_loss_coef == 0.0 and self.z_loss_coef == 0.0:
            return torch.zeros((), device=router_logits.device, dtype=router_logits.dtype)

        tokens_local = int(router_probs.shape[0])

        # Switch-style load balancing (top-1): E * sum(importance * load).
        #
        # `importance` must keep gradients (comes from router softmax).
        # `load` comes from argmax routing and has no gradients.
        load_balance_loss = torch.zeros((), device=router_logits.device, dtype=torch.float32)
        if self.aux_loss_coef != 0.0:
            importance = router_probs.mean(dim=0)  # [E], grad flows to router
            load_counts = torch.bincount(top1_idx, minlength=self.num_experts).to(dtype=torch.float32)  # [E], no grad
            tokens = float(tokens_local)

            if self._ep_enabled and dist.is_available() and dist.is_initialized():
                # EP batches can be imbalanced across ranks; estimate `load` globally for stability.
                t = torch.tensor([tokens_local], device=router_probs.device, dtype=torch.int64)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                tokens = float(max(int(t.item()), 1))
                dist.all_reduce(load_counts, op=dist.ReduceOp.SUM)

            load = load_counts / tokens  # [E]
            load_balance_loss = (importance * load).sum() * float(self.num_experts)
        z_loss = router_logits.float().logsumexp(dim=-1).pow(2).mean()
        return (self.aux_loss_coef * load_balance_loss + self.z_loss_coef * z_loss).to(dtype=router_logits.dtype)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        b, s, d = x.shape
        tokens = b * s
        x_flat = x.reshape(tokens, d)

        router_logits = self.router(x_flat)  # [T, E]
        if self.router_jitter:
            router_logits = router_logits + self.router_jitter * torch.randn_like(router_logits)

        router_probs = torch.softmax(router_logits.float(), dim=-1)  # [T, E]
        if self.top_k == 1:
            gate, expert_idx = router_probs.max(dim=-1)  # [T]
            aux_loss = self._aux_loss(router_logits, router_probs, expert_idx)
        else:
            topk_probs, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1)  # [T, K]
            if self.normalize_gates:
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            gate = topk_probs.reshape(-1)  # [T*K]
            expert_idx = topk_idx.reshape(-1)  # [T*K]
            aux_loss = self._aux_loss(router_logits, router_probs, topk_idx[:, 0])

        if not self._ep_enabled:
            tokens_per_expert = tokens if self.top_k == 1 else tokens * self.top_k
            capacity = int(math.ceil(self.capacity_factor * (tokens_per_expert / self.num_experts)))
            capacity = max(1, capacity)

            if self.top_k == 1:
                x_rep = x_flat
            else:
                x_rep = x_flat.repeat_interleave(self.top_k, dim=0)  # [T*K, D]

            order = torch.argsort(expert_idx)
            x_sorted = x_rep[order]
            gate_sorted = gate[order].to(dtype=x.dtype)
            expert_sorted = expert_idx[order]

            out_sorted = torch.zeros_like(x_sorted)
            counts = torch.bincount(expert_sorted, minlength=self.num_experts)
            offsets = torch.cumsum(counts, dim=0)

            start = 0
            for e in range(self.num_experts):
                end = int(offsets[e].item())
                n = end - start
                if n <= 0:
                    start = end
                    continue
                take = min(n, capacity)
                if take > 0:
                    y = self.experts[e](x_sorted[start : start + take])
                    y = y * gate_sorted[start : start + take].unsqueeze(-1)
                    out_sorted[start : start + take] = y
                start = end

            inv = torch.empty_like(order)
            inv[order] = torch.arange(order.numel(), device=order.device)
            out = out_sorted[inv]
            if self.top_k == 1:
                return out.reshape(b, s, d), aux_loss.to(dtype=x.dtype)
            out = out.reshape(tokens, self.top_k, d).sum(dim=1)
            return out.reshape(b, s, d), aux_loss.to(dtype=x.dtype)

        num_local = self._num_local_experts
        dst_rank = (expert_idx // num_local).to(torch.int64)
        local_expert = (expert_idx % num_local).to(torch.int64)

        # Capacity is defined in terms of tokens-per-expert across the *global* batch.
        tok = torch.tensor([tokens], device=x.device, dtype=torch.int64)
        dist.all_reduce(tok, op=dist.ReduceOp.SUM)
        tokens_global = int(tok.item())
        capacity = int(math.ceil(self.capacity_factor * (tokens_global / self.num_experts)))
        capacity = max(1, capacity)

        perm = torch.argsort(dst_rank)
        dst_rank_sorted = dst_rank[perm]

        send_counts = torch.bincount(dst_rank_sorted, minlength=self._ep_world_size).to(torch.int64)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        recv_total = int(recv_counts.sum().item())
        x_send = x_flat[perm]
        gate_send = gate[perm].to(dtype=x.dtype)
        local_expert_send = local_expert[perm].to(dtype=x.dtype)
        x_send_aug = torch.cat([x_send, gate_send.unsqueeze(-1), local_expert_send.unsqueeze(-1)], dim=-1)  # [T, D+2]
        x_recv_aug = torch.empty((recv_total, d + 2), device=x.device, dtype=x.dtype)

        dist.all_to_all_single(
            x_recv_aug,
            x_send_aug,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
        )

        x_recv = x_recv_aug[:, :d]
        gate_recv = x_recv_aug[:, d]
        local_expert_recv = x_recv_aug[:, d + 1].to(torch.int64)

        y_recv = torch.zeros_like(x_recv)
        order_e = torch.argsort(local_expert_recv)
        x_e = x_recv[order_e]
        gate_e = gate_recv[order_e]
        exp_e = local_expert_recv[order_e]

        counts_e = torch.bincount(exp_e, minlength=num_local).to(torch.int64)
        offsets_e = torch.cumsum(counts_e, dim=0)
        start = 0
        for e in range(num_local):
            end = int(offsets_e[e].item())
            n = end - start
            if n <= 0:
                start = end
                continue
            take = min(n, capacity)
            if take > 0:
                yy = self.experts[e](x_e[start : start + take])
                yy = yy * gate_e[start : start + take].unsqueeze(-1)
                y_recv[order_e[start : start + take]] = yy
            start = end

        y_back = torch.empty((tokens, d), device=x.device, dtype=x.dtype)

        dist.all_to_all_single(
            y_back,
            y_recv,
            output_split_sizes=send_counts.tolist(),
            input_split_sizes=recv_counts.tolist(),
        )

        out = torch.zeros((tokens, d), device=x.device, dtype=x.dtype)
        # `y_back` is in `perm` order (same as `x_send`), so `perm` is sufficient to unpermute.
        out[perm] = y_back
        return out.reshape(b, s, d), aux_loss.to(dtype=x.dtype)
