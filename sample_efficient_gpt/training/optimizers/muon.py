import torch
from torch import Tensor
import torch.cuda.nvtx as nvtx

# from sample_efficient_gpt.training.optimizers.muon_triton import newton_schulz_triton

# -----------------------------------------------------------------------------
# Muon optimizer


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @nvtx.range("muon step")
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            momentum = group["momentum"]
            for p in params:
                grad = p.grad

                # out_dim / inp_dim (reversed order bc in our implementation of Linear dims are swapped)
                mup_mult = max(1, p.size(-2) / p.size(-1)) ** 0.5
                # variance preserving multiplier TODO: try on longer runs
                # var_preserving_multiplier = 0.2 * max(p.size(-2), p.size(-1)) ** 0.5
                var_preserving_multiplier = 1.0
                eff_lr = group["lr"] * mup_mult * var_preserving_multiplier * getattr(p, "lr_mul", 1.0)
                eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum_buffer = state["momentum_buffer"]

                # apply wd
                p.mul_(1 - eff_weight_decay)
                # interpolate momentum
                momentum_buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(momentum_buffer, momentum)
                # v = newton_schulz_triton(grad.bfloat16(), 5)
                v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                p.add_(other=v, alpha=-eff_lr)
