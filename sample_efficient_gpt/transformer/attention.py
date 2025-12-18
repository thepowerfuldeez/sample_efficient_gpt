import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

import os
from einops import einsum, rearrange

from sample_efficient_gpt.transformer.core import Linear, softmax
from sample_efficient_gpt.transformer.rope import RotatyPositionalEmbedding
from sample_efficient_gpt.transformer.ops.triton_flash_attn import TritonFlashAttnFunc
from sample_efficient_gpt.utils.profiling import nvtx_range


class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None

    def append(self, k: Float[Tensor, "bs seq d_model"], v: Float[Tensor, "bs seq d_model"]):
        if self.keys is None:
            self.keys = k
            self.values = v
        else:
            self.keys = torch.cat((self.keys, k), dim=1)
            self.values = torch.cat((self.values, v), dim=1)
        return self.keys, self.values


class SelfDotProductAttnQKNorm(nn.Module):
    def __init__(
        self,
        head_dim: int,
    ):
        super().__init__()
        # We apply per-token RMS normalization (not L2) to Q and K, then use the same Triton
        # attention kernel as the baseline path (so train/eval stay consistent). A scalar gain
        # is provided so conversions can tune the initial logit scale.
        #
        # With typical muP-ish init, RMS(Q) and RMS(K) are ~1, so gain=1 is near a no-op.
        self.gain = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(
        self,
        Q: Float[Tensor, "... seq_len d_k"],
        K: Float[Tensor, "... seq_len d_k"],
        V: Float[Tensor, "... seq_len d_k"],
        is_causal: bool = True,
        q_start: int = 0,
    ) -> Float[Tensor, "... seq_len d_k"]:
        eps = 1e-6
        with torch.autocast("cuda", enabled=False):
            q_f32 = Q.to(torch.float32)
            k_f32 = K.to(torch.float32)

            q_inv = torch.rsqrt((q_f32 * q_f32).mean(dim=-1, keepdim=True) + eps)
            k_inv = torch.rsqrt((k_f32 * k_f32).mean(dim=-1, keepdim=True) + eps)

            q_norm = (q_f32 * q_inv).to(Q.dtype)
            k_norm = (k_f32 * k_inv).to(K.dtype)

        q_norm = (q_norm * self.gain.to(device=q_norm.device, dtype=q_norm.dtype)).contiguous()
        k_norm = k_norm.contiguous()
        qknorm_attn_impl = os.environ.get("SEGPT_QKNORM_ATTN_IMPL", "sdpa").lower()
        if qknorm_attn_impl == "triton":
            # torch.autograd.Function.apply does not accept keyword arguments.
            return TritonFlashAttnFunc.apply(q_norm, k_norm, V, q_start, is_causal)

        # PyTorch SDPA is numerically stable in bf16/fp16 and supports flash-attn backends.
        # When using KV-cache decoding, `q_start` shifts the causal diagonal; build an explicit
        # mask (typically tiny, since seq_len is often 1 at decode time).
        if is_causal and q_start != 0:
            q_len = q_norm.size(-2)
            k_len = k_norm.size(-2)
            q_idx = q_start + torch.arange(q_len, device=q_norm.device)
            k_idx = torch.arange(k_len, device=q_norm.device)
            attn_mask = k_idx[None, :] <= q_idx[:, None]
            return torch.nn.functional.scaled_dot_product_attention(q_norm, k_norm, V, attn_mask=attn_mask)

        return torch.nn.functional.scaled_dot_product_attention(q_norm, k_norm, V, is_causal=is_causal)


# Modification using regular RMSNorm with sqrt(d) scaling
# class SelfDotProductAttnQKNorm(nn.Module):
#     def __init__(
#         self,
#         head_dim: int,
#     ):
#         super().__init__()
#         self.q_norm = RMSNorm(head_dim)
#         self.k_norm = RMSNorm(head_dim)

#     def forward(
#         self,
#         q: Float[Tensor, "... seq_len d_k"],
#         k: Float[Tensor, "... seq_len d_k"],
#         v: Float[Tensor, "... seq_len d_k"],
#         mask: Float[Tensor, "... seq_len seq_len"] | None = None,
#     ) -> Float[Tensor, "... seq_len d_k"]:
#         with torch.autocast("cuda", enabled=False):
#             q_norm = self.q_norm(q)
#             k_norm = self.k_norm(k)
#             attn_scores = einsum(q_norm, k_norm, "... s1 d_k, ... s2 d_k -> ... s1 s2")
#             attn_scores *= torch.rsqrt(torch.tensor(q.size(-1)))
#             if mask is not None:
#                 attn_scores.masked_fill_(~mask, float("-inf"))
#             probs = softmax(attn_scores, -1)
#         return probs @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        theta: float = 10_000,
        max_seq_len: int = 4096,
        qknorm: bool = False,
        value_residual: bool = False,
        gating: bool = False,
        rope_interleaved: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "Hidden dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.n_kv_heads = int(n_kv_heads) if n_kv_heads is not None else n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"Invalid GQA: n_heads={self.n_heads} not divisible by n_kv_heads={self.n_kv_heads}")
        self.rope = RotatyPositionalEmbedding(
            theta,
            d_k=d_model // n_heads,
            max_seq_len=max_seq_len,
            device=device,
            rope_interleaved=rope_interleaved,
        )
        head_dim = d_model // n_heads
        q_out = n_heads * head_dim
        kv_out = self.n_kv_heads * head_dim
        self.qkv = Linear(d_model, q_out + 2 * kv_out, device=device, dtype=dtype)
        self.out = Linear(d_model, d_model, device=device, dtype=dtype)
        self.qknorm = qknorm
        if self.qknorm:
            self.sdpa_qknorm = SelfDotProductAttnQKNorm(head_dim)
        if value_residual:
            # Initialize to a no-op for V when v1 is not provided (common at eval):
            # V = scale * (alpha1*V + alpha2*V1) / sqrt(alpha1^2 + alpha2^2)
            # With alpha1=1, alpha2=0, scale=1 -> V unchanged.
            self.alpha1, self.alpha2 = (
                nn.Parameter(torch.tensor([1.0], device=device)),
                nn.Parameter(torch.tensor([0.0], device=device)),
            )
            self.scale = nn.Parameter(torch.tensor([1.0], device=device))
        else:
            # Keep these on the module's device and visible to torch.compile/DDP by
            # registering them as buffers (not plain Tensor attributes).
            self.register_buffer("alpha1", torch.tensor([1.0], device=device), persistent=False)
            self.register_buffer("alpha2", torch.tensor([0.0], device=device), persistent=False)
            self.register_buffer("scale", torch.tensor([1.0], device=device), persistent=False)

        # elementwise SDPA gating on attention after concat
        self.gating = gating
        if self.gating:
            if self.gating == "elementwise":
                self.attn_gate = Linear(d_model, d_model, device=device, dtype=dtype)
            elif self.gating == "per-head":
                # gating single scalar per head
                self.attn_gate = Linear(d_model, n_heads, device=device, dtype=dtype)
            elif self.gating == "per-head-hd":
                # gating single scalar per head
                self.attn_gate = Linear(d_model // n_heads, n_heads, device=device, dtype=dtype)
            else:
                raise ValueError(f"{self.gating=} is undefined")

    @nvtx_range("attention")
    def forward(
        self,
        x: Float[Tensor, "b seq d"],
        token_positions: Int[Tensor, "b seq"] | None = None,
        v1: Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> Float[Tensor, "b seq d"]:
        head_dim = x.shape[-1] // self.n_heads
        q_out = self.n_heads * head_dim
        kv_out = self.n_kv_heads * head_dim
        proj = self.qkv(x)
        Q = proj[..., :q_out]
        K = proj[..., q_out : q_out + kv_out]
        V = proj[..., q_out + kv_out :]
        seq_len = Q.size(1)
        if kv_cache is not None:
            past = kv_cache.keys.size(1) if kv_cache.keys is not None else 0
            # we need to adjust position for Q when K seq is 1
            pos = torch.arange(past, past + seq_len, device=Q.device, dtype=torch.long)
            q_start = past
        else:
            pos = None
            q_start = 0
        Q = rearrange(Q, "b seq (h head_d) -> (h b) seq head_d", h=self.n_heads)
        K = rearrange(K, "b seq (h head_d) -> (h b) seq head_d", h=self.n_kv_heads)

        with torch.autocast("cuda", enabled=False), nvtx_range("RoPE"):
            Q = self.rope(Q, pos)
            K = self.rope(K, pos)

        # Expand KV heads to match Q heads (GQA -> MHA layout expected by kernels).
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            K = rearrange(K, "(h b) seq head_d -> h b seq head_d", h=self.n_kv_heads)
            K = K.repeat_interleave(repeat, dim=0)
            K = rearrange(K, "h b seq head_d -> (h b) seq head_d", h=self.n_heads)

        V = rearrange(V, "b seq (h head_d) -> (h b) seq head_d", h=self.n_kv_heads).contiguous()
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            V = rearrange(V, "(h b) seq head_d -> h b seq head_d", h=self.n_kv_heads)
            V = V.repeat_interleave(repeat, dim=0)
            V = rearrange(V, "h b seq head_d -> (h b) seq head_d", h=self.n_heads)

        if kv_cache is not None:
            K, V = kv_cache.append(K, V)
        if v1 is None:
            V1 = V
        else:
            V1 = v1.view_as(V).contiguous()

        # value residual learning
        V = self.scale * (self.alpha1 * V + self.alpha2 * V1) * torch.rsqrt(self.alpha1**2 + self.alpha2**2 + 1e-8)
        assert V1.is_contiguous()
        assert V.is_contiguous()
        if self.qknorm:
            attn: Float[Tensor, "(h b) seq head_d"] = self.sdpa_qknorm(Q, K, V, True, q_start=q_start)
        else:
            attn_impl = os.environ.get("SEGPT_ATTN_IMPL", "sdpa").lower()
            if attn_impl == "sdpa":
                attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
            else:
                # torch.autograd.Function.apply does not accept keyword arguments.
                # TritonFlashAttnFunc.forward signature is (Q, K, V, q_start=0, is_causal=True, ...)
                attn = TritonFlashAttnFunc.apply(Q, K, V, q_start, True)

        if not self.gating:
            attn_cat = rearrange(attn, "(h b) seq head_d -> b seq (h head_d)", h=self.n_heads)
        else:
            if self.gating == "elementwise":
                attn_cat = rearrange(attn, "(h b) seq head_d -> b seq (h head_d)", h=self.n_heads)
                # apply gating after concat and before out
                attn_cat = (2.0 * self.attn_gate(x).sigmoid()) * attn_cat
            elif self.gating == "per-head":
                # apply gating (single scalar per each head)
                gate: Float[Tensor, "b seq h 1"] = (2.0 * self.attn_gate(x).sigmoid()).unsqueeze(-1)
                attn: Float[Tensor, "b seq h head_d"] = rearrange(
                    attn, "(h b) seq head_d -> b seq h head_d", h=self.n_heads
                )
                attn_cat = rearrange(gate * attn, "b seq h head_d -> b seq (h head_d)", h=self.n_heads)
            elif self.gating == "per-head-hd":
                # apply gating (single scalar per each head)
                head_dim = Q.shape[-1]
                gate: Float[Tensor, "b seq h 1"] = (2.0 * self.attn_gate(x[..., :head_dim]).sigmoid()).unsqueeze(-1)
                attn: Float[Tensor, "b seq h head_d"] = rearrange(
                    attn, "(h b) seq head_d -> b seq h head_d", h=self.n_heads
                )
                attn_cat = rearrange(gate * attn, "b seq h head_d -> b seq (h head_d)", h=self.n_heads)

        attn_out = self.out(attn_cat)
        # pass current value vector
        return attn_out, V


if __name__ == "__main__":
    sa = MultiHeadSelfAttention(1024, 8)
    x = torch.randn(4, 64, 1024)
    token_positions = torch.arange(0, 64).unsqueeze(0)
    sa(x, token_positions)
