"""Rotary Positional Embeddings"""

import time

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange


def get_cos_sin(
    max_seq_len: int, theta_base: float, d_k: int, device
) -> tuple[Float[Tensor, "max_seq d_k // 2"], Float[Tensor, "max_seq d_k // 2"]]:
    """
    Get cos and sin for every position
    """
    # for i = 1 (second sequence)
    thetas = torch.tensor(theta_base, device=device).unsqueeze(0).repeat(d_k // 2)
    j = torch.arange(0, d_k // 2, device=device)
    inv_freqs: Float[Tensor, d_k // 2] = theta_base ** (-2 * j / d_k)
    thetas: Float[Tensor, max_seq_len, d_k // 2] = torch.outer(torch.arange(max_seq_len, device=device), inv_freqs)
    return thetas.cos(), thetas.sin()


class RotatyPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        time.monotonic()
        cos, sin = get_cos_sin(max_seq_len, theta, d_k, device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # logger.info(f"RoPE initialized in {time.monotonic() - t0:.2}s.")

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_k"]:
        in_dtype = x.dtype
        if token_positions is not None:
            cos: Float[Tensor, "seq_len d_k // 2"] = self.cos[token_positions]
            sin: Float[Tensor, "seq_len d_k // 2"] = self.sin[token_positions]
        else:
            cos: Float[Tensor, "seq_len d_k // 2"] = self.cos[: x.size(-2)]
            sin: Float[Tensor, "seq_len d_k // 2"] = self.sin[: x.size(-2)]

        x_pairs = rearrange(x.to(torch.float32), "... seq_len (d_k_half t) -> ... seq_len d_k_half t", t=2)
        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]

        row1 = x1 * cos - x2 * sin
        row2 = x1 * sin + x2 * cos
        rotated = torch.stack([row1, row2], dim=-1)
        rotated = rearrange(rotated, "... seq_len d_k_half t -> ... seq_len (d_k_half t)", t=2)
        return rotated.to(in_dtype)


if __name__ == "__main__":
    rot = RotatyPositionalEmbedding(10, 32, 16)
