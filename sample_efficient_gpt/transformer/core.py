import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum

from sample_efficient_gpt.utils.profiling import nvtx_range


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        in_features: final dim of the input
        out_features: final dim of the output
        """
        super().__init__()
        weight: Tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.weight: Float[Tensor, "out_features in_features"] = nn.Parameter(weight)
        # std: float = 2.0 / (in_features + out_features)
        # in muP we use different parametrization
        sigma: float = 1.0 / (in_features**0.5)
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3 * sigma, b=3 * sigma)

    @nvtx_range("linear forward")
    def forward(self, x: Float[Tensor, "batch ... in_features"]) -> Float[Tensor, "batch ... out_features"]:
        # x is row-wise vector
        out: Float[Tensor, "batch ... out_features"] = einsum(
            x,
            self.weight,
            "batch ... in_features, out_features in_features -> batch ... out_features",
        )
        return out


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()
        weight: Tensor = torch.empty(vocab_size, d_model, device=device, dtype=dtype)
        self.weight: Float[Tensor, "vocab_size d_model"] = nn.Parameter(weight)
        # init std per muP
        sigma = 1 / (d_model**0.5)
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_model"]:
        return torch.nn.functional.embedding(x, self.weight)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, position: int | None = None, device=None, dtype=None):
        """
        LayerNorm scaling - scale by the rsqrt of position
        """
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        if position is None:
            scale = torch.tensor(1.0, device=device, dtype=torch.float32)
        else:
            # rsqrt(position) as float tensor
            scale = torch.rsqrt(torch.tensor(float(position), device=device, dtype=torch.float32))
        self.register_buffer("position_scale", scale, persistent=False)

    @nvtx_range("rmsnorm forward")
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        x is an activation from residual stream
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        with torch.autocast("cuda", enabled=False):
            reverse_rms: Float[Tensor, "... 1"] = torch.rsqrt((x * x).mean(-1) + self.eps).unsqueeze(-1)
            out: Tensor = x * reverse_rms * self.gain
        # apply layernorm scaling if enabled
        out.mul_(self.position_scale)
        return out.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # we would use larger matrix and split it by 2 later
        # round to 64 to use hardware better
        # d_ff: int = int(d_model * d_mult // 64 * 64)
        d_ff: int = int(d_ff // 64 * 64)
        self.up = Linear(d_model, d_ff * 2, device, dtype)
        self.down = Linear(d_ff, d_model, device, dtype)

    @nvtx_range("swiglu forward")
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        projected: Float[Tensor, "... 2*d_ff"] = self.up(x)
        left, right = projected.chunk(2, dim=-1)
        return self.down(left * torch.sigmoid(left) * right)


def softmax(x: Tensor, dim: int = 0, temperature: float = 1.0) -> Tensor:
    o: Tensor = x - x.max(dim=dim, keepdim=True)[0]
    assert temperature > 0, "temperature must be more than 0"
    if temperature != 1.0:
        o /= temperature
    return o.exp() / o.exp().sum(dim=dim, keepdim=True)
