import torch
import torch.nn as nn
from torch import Tensor

from sample_efficient_gpt.transformer.core import SwiGLU


class _ExpertsParams(nn.Module):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        *,
        add_bias: bool,
        std: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features, device=device, dtype=dtype))
        self.bias = None
        if add_bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features, device=device, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            self.bias.zero_()


class _SonicMoEParams(nn.Module):
    def __init__(
        self,
        num_experts: int,
        *,
        hidden_size: int,
        intermediate_size: int,
        add_bias: bool,
        std: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.c_fc = _ExpertsParams(
            num_experts,
            hidden_size,
            2 * intermediate_size,
            add_bias=add_bias,
            std=std,
            device=device,
            dtype=dtype,
        )
        self.c_proj = _ExpertsParams(
            num_experts,
            intermediate_size,
            hidden_size,
            add_bias=add_bias,
            std=std,
            device=device,
            dtype=dtype,
        )


class SonicMoEAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        aux_loss_coef: float,
        num_shared_experts: int = 0,
        add_bias: bool = False,
        std: float = 0.02,
        kernel_backend: str = "sonicmoe",
        device=None,
        dtype=None,
    ):
        super().__init__()
        del kernel_backend

        self.aux_loss_coef = float(aux_loss_coef)
        self.num_shared_experts = int(num_shared_experts)
        if self.num_shared_experts < 0 or self.num_shared_experts >= int(num_experts):
            raise ValueError("num_shared_experts must be in [0, num_experts)")
        self.num_routed_experts = int(num_experts) - self.num_shared_experts
        if self.num_routed_experts <= 0:
            raise ValueError("num_routed_experts must be > 0")

        self._use_fake = not torch.cuda.is_available()
        self._moe_dtype = torch.float16 if not self._use_fake else (dtype or torch.float32)
        shared_dtype = self._moe_dtype if not self._use_fake else dtype
        self.shared_experts = nn.ModuleList(
            [SwiGLU(d_model, d_ff, device=device, dtype=shared_dtype) for _ in range(self.num_shared_experts)]
        )
        if self._use_fake:
            self._kernel_backend = None
            self.moe = _SonicMoEParams(
                self.num_routed_experts,
                hidden_size=int(d_model),
                intermediate_size=int(d_ff),
                add_bias=bool(add_bias),
                std=float(std),
                device=device,
                dtype=dtype,
            )
            return

        try:
            from sonicmoe import KernelBackendMoE, MoE  # type: ignore[import-not-found]
            from sonicmoe.enums import ActivationType
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Failed to import `sonicmoe`.\n"
                "If you added the submodule, ensure it is initialized and installed, e.g.:\n"
                "  git submodule update --init --recursive\n"
                "  uv pip install -e ext/sonic-moe\n"
                "and install its prerequisites (see ext/sonic-moe/README.md)."
            ) from e

        self._kernel_backend = KernelBackendMoE.sonicmoe

        moe = MoE(
            num_experts=int(self.num_routed_experts),
            num_experts_per_tok=int(top_k),
            hidden_size=int(d_model),
            intermediate_size=int(d_ff),
            activation_function=ActivationType.SWIGLU,  # SwiGLU activation
            add_bias=bool(add_bias),
            std=float(std),  # Weight initialization std
        )
        if device is not None or self._moe_dtype is not None:
            moe = moe.to(device=device, dtype=self._moe_dtype)
        self.moe = moe

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self._use_fake:
            raise RuntimeError("SonicMoEAdapter forward requires CUDA.")

        b, s, d = x.shape
        with torch.autocast("cuda", enabled=False):
            x_moe = x if x.dtype == self._moe_dtype else x.to(self._moe_dtype)
            shared_out = None
            if self.num_shared_experts > 0:
                shared_out = torch.zeros_like(x_moe)
                for expert in self.shared_experts:
                    shared_out = shared_out + expert(x_moe)
                if self.num_shared_experts > 1:
                    shared_out = shared_out / float(self.num_shared_experts)

            y, aux = self.moe(x_moe.reshape(b * s, d), kernel_backend_moe=self._kernel_backend)
            y = y.reshape(b, s, d)
            if shared_out is not None:
                y = y + shared_out

        if y.dtype != x.dtype:
            y = y.to(dtype=x.dtype)
        aux = aux * self.aux_loss_coef
        return y, aux.to(dtype=x.dtype)
