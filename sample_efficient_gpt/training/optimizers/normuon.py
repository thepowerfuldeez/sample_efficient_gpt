import torch
import torch.distributed as dist
from torch import Tensor
from typing import Any

from sample_efficient_gpt.training.optimizers.muon_triton import newton_schulz_triton
from sample_efficient_gpt.utils.profiling import nvtx_range

# -----------------------------------------------------------------------------
# Muon optimizer


def normuon_update(p, v, second_momentum_buffer, eff_lr, eff_weight_decay, beta2):
    ###################################
    v_norm = v.norm(dim=(-2, -1), keepdim=True)
    v_mean = (
        torch.mean(v * v, dim=-1, keepdim=True) if p.size(-2) >= p.size(-1) else torch.mean(v * v, dim=-2, keepdim=True)
    )
    second_momentum_buffer.lerp_(v_mean, 1 - beta2)
    step_size = torch.rsqrt(second_momentum_buffer.clamp_min(1e-10))
    v.mul_(step_size)
    v_norm_new = v.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10)
    v.mul_(v_norm / v_norm_new)
    ####################################

    cautious_mask = torch.eq(torch.signbit(v), torch.signbit(p))
    p.addcmul_(p, cautious_mask.to(p.dtype), value=-eff_weight_decay)

    p.add_(other=v, alpha=-eff_lr)
    return v_norm.squeeze()


class Muon(torch.optim.Optimizer):
    """
    NorMuon - a variant of the Muon optimizer that introduces neuron-wise normalization to improve stability and convergence efficiency.

    https://arxiv.org/abs/2510.05491

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, beta2=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            print(f"Grouping {len(group_params)} params of shape {size}")
            param_groups.append(dict(params=group_params))

        self.is_distributed = False
        # self.is_distributed = dist.is_initialized()
        # self.world = dist.get_world_size() if self.is_distributed else 1
        super().__init__(param_groups, defaults)

    @nvtx_range("muon step")
    @torch.no_grad()
    def step(self, closure: Any | None = None):
        update_rms = torch.tensor(0.0, dtype=torch.float32, device=self.param_groups[0]["params"][0].device)
        n = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            momentum = group["momentum"]
            beta2 = group["beta2"]
            for p in params:
                grad = p.grad
                if grad is None:
                    continue

                # out_dim / inp_dim (reversed order bc in our implementation of Linear dims are swapped)
                mup_mult = max(1, p.size(0) / p.size(1)) ** 0.5
                # variance preserving multiplier TODO: try on longer runs
                # var_preserving_multiplier = 0.2 * max(p.size(-2), p.size(-1)) ** 0.5
                var_preserving_multiplier = 1.0
                assert hasattr(p, "lr_mul")
                eff_lr = group["lr"] * mup_mult * var_preserving_multiplier * getattr(p, "lr_mul", 1.0)

                # decoupled wd
                eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)

                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    state["second_momentum_buffer"] = (
                        torch.zeros_like(grad[..., 0:1])
                        if p.size(-2) >= p.size(-1)
                        else torch.zeros_like(grad[0:1, ...])
                    )
                momentum_buffer = state["momentum_buffer"]
                second_momentum_buffer = state["second_momentum_buffer"]

                # interpolate momentum
                momentum_buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(momentum_buffer, momentum)

                v = newton_schulz_triton(grad.bfloat16()).float()

                v_norm = normuon_update(p, v, second_momentum_buffer, eff_lr, eff_weight_decay, beta2)
                # v_norm = normuon_triton_update(p, v, second_momentum_buffer, eff_lr, eff_weight_decay, beta2)
                update_rms += v_norm * v_norm
                n += 1
        return update_rms.sqrt() / (n + 1e-10)
