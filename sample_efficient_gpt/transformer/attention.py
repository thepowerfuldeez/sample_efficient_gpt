import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from einops import einsum, rearrange

from sample_efficient_gpt.transformer.core import Linear, softmax
from sample_efficient_gpt.transformer.rope import RotatyPositionalEmbedding
from sample_efficient_gpt.transformer.triton_flash_attn import TritonFlashAttnFunc
from sample_efficient_gpt.transformer.triton_flash_attn_qknorm import TritonFlashAttnQKNormFunc
from sample_efficient_gpt.utils.profiling import nvtx_range


class SelfDotProductAttnQKNorm(nn.Module):
    def __init__(
        self,
        context_length: int,
    ):
        super().__init__()
        self.gain = nn.Parameter(torch.log2(torch.tensor(context_length**2 - context_length)))

    def forward(
        self,
        Q: Float[Tensor, "... seq_len d_k"],
        K: Float[Tensor, "... seq_len d_k"],
        V: Float[Tensor, "... seq_len d_k"],
        is_causal: bool = True,
    ) -> Float[Tensor, "... seq_len d_k"]:
        return TritonFlashAttnQKNormFunc.apply(Q, K, V, self.gain, is_causal)


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
        theta: float = 10_000,
        max_seq_len: int = 4096,
        qknorm: bool = False,
        value_residual: bool = False,
        gating: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "Hidden dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.rope = RotatyPositionalEmbedding(theta, d_k=d_model // n_heads, max_seq_len=max_seq_len, device=device)
        self.qkv = Linear(d_model, d_model * 3, device=device, dtype=dtype)
        self.out = Linear(d_model, d_model, device=device, dtype=dtype)
        self.qknorm = qknorm
        if self.qknorm:
            self.sdpa_qknorm = SelfDotProductAttnQKNorm(d_model // n_heads)
        if value_residual:
            self.alpha1, self.alpha2 = (
                nn.Parameter(torch.tensor(0.5, device=device)),
                nn.Parameter(torch.tensor(0.5, device=device)),
            )
            self.scale = nn.Parameter(torch.tensor(1.0, device=device))
        else:
            self.alpha1, self.alpha2, self.scale = (
                torch.tensor(1.0, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(1.0, device=device),
            )

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
    ) -> Float[Tensor, "b seq d"]:
        Q, K, V = self.qkv(x).chunk(3, -1)
        seq_len = Q.size(1)
        Q = rearrange(Q, "b seq (h head_d) -> (h b) seq head_d", h=self.n_heads)
        K = rearrange(K, "b seq (h head_d) -> (h b) seq head_d", h=self.n_heads)

        with torch.autocast("cuda", enabled=False), nvtx_range("RoPE"):
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        V = rearrange(V, "b seq (h head_d) -> (h b) seq head_d", h=self.n_heads).contiguous()
        if v1 is None:
            V1 = V
        else:
            V1 = v1.view_as(V).contiguous()

        # value residual learning
        V = self.scale * (self.alpha1 * V + self.alpha2 * V1) * torch.rsqrt(self.alpha1**2 + self.alpha2**2 + 1e-8)
        assert V1.is_contiguous()
        assert V.is_contiguous()
        if self.qknorm:
            attn: Float[Tensor, "(h b) seq head_d"] = self.sdpa_qknorm(Q, K, V, True)
        else:
            attn: Float[Tensor, "(h b) seq head_d"] = TritonFlashAttnFunc.apply(Q, K, V, True)

        if not self.gating:
            attn_cat = rearrange(attn, "(h b) seq head_d -> b seq (h head_d)", h=self.n_heads)
        else:
            if self.gating == "elementwise":
                attn_cat = rearrange(attn, "(h b) seq head_d -> b seq (h head_d)", h=self.n_heads)
                # apply gating after concat and before out
                attn_cat = self.attn_gate(x).sigmoid() * attn_cat
            elif self.gating == "per-head":
                # apply gating (single scalar per each head)
                gate: Float[Tensor, "b seq h 1"] = self.attn_gate(x).sigmoid().unsqueeze(-1)
                attn: Float[Tensor, "b seq h head_d"] = rearrange(
                    attn, "(h b) seq head_d -> b seq h head_d", h=self.n_heads
                )
                attn_cat = rearrange(gate * attn, "b seq h head_d -> b seq (h head_d)", h=self.n_heads)
            elif self.gating == "per-head-hd":
                # apply gating (single scalar per each head)
                head_dim = Q.shape[-1]
                gate: Float[Tensor, "b seq h 1"] = self.attn_gate(x[..., :head_dim]).sigmoid().unsqueeze(-1)
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
