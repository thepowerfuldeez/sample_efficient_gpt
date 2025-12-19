import torch
import torch.nn as nn
from torch import Tensor


class SonicMoEAdapter(nn.Module):
    """
    Thin adapter around Dao-AILab/sonic-moe to match this codebase's MoE interface.

    - Input/Output: [B, S, D]
    - Returns: (y, aux_loss) where aux_loss is already scaled by `aux_loss_coef`.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        aux_loss_coef: float,
        add_bias: bool = False,
        std: float = 0.02,
        kernel_backend: str = "sonicmoe",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if device is not None and str(device) != "cuda":
            raise ValueError("SonicMoEAdapter currently requires device='cuda'.")

        try:
            from sonicmoe import KernelBackendMoE, MoE  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Failed to import `sonicmoe`.\n"
                "If you added the submodule, ensure it is initialized and installed, e.g.:\n"
                "  git submodule update --init --recursive\n"
                "  uv pip install -e ext/sonic-moe\n"
                "and install its prerequisites (see ext/sonic-moe/README.md)."
            ) from e

        self.aux_loss_coef = float(aux_loss_coef)
        if kernel_backend == "sonicmoe":
            self._kernel_backend = KernelBackendMoE.sonicmoe
        elif kernel_backend == "torch":
            self._kernel_backend = KernelBackendMoE.torch
        elif kernel_backend == "scattermoe":
            self._kernel_backend = KernelBackendMoE.scattermoe
        else:
            raise ValueError(f"Unknown sonicmoe kernel backend: {kernel_backend!r}")

        moe = MoE(
            num_experts=int(num_experts),
            num_experts_per_tok=int(top_k),
            hidden_size=int(d_model),
            intermediate_size=int(d_ff),
            is_glu=True,
            add_bias=bool(add_bias),
            std=float(std),
        )
        if device is not None or dtype is not None:
            moe = moe.to(device=device, dtype=dtype)
        self.moe = moe

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        b, s, d = x.shape
        y, aux = self.moe(x.reshape(b * s, d), kernel_backend_moe=self._kernel_backend)
        aux = aux * self.aux_loss_coef
        return y.reshape(b, s, d), aux.to(dtype=x.dtype)

