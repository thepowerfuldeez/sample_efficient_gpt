import torch
from torch import nn
import torch.nn.functional as F
import time
from collections.abc import Callable

from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.config import Float8LinearConfig, Float8LinearRecipeName, Float8GemmConfig
from torchao.float8.float8_linear import (
    LinearMMConfig,
    ScaledMMConfig,
    matmul_with_hp_or_float8_args,
    ScalingType,
    WeightWithDynamicFloat8CastTensor,
)
from torchao.float8 import convert_to_float8_training

from sample_efficient_gpt.transformer.core import Linear
from sample_efficient_gpt.train import apply_overrides, cfg, Trainer


config = Float8LinearConfig(
    pad_inner_dim=True,
    # enable_fsdp_float8_all_gather=True
    gemm_config_grad_input=Float8GemmConfig(use_fast_accum=True),
    gemm_config_grad_weight=Float8GemmConfig(use_fast_accum=True),
    gemm_config_output=Float8GemmConfig(use_fast_accum=True),
)


class Float8Linear(Linear):
    """
    Note: this is **not** a public API and is only intended to be used
    inside of this repository. Please file an issue if you would benefit
    from this being a public API.

    A wrapper around a `torch.nn.Linear` module which does fp8 compute.
    """

    def __init__(self, *args, **kwargs):
        """
        Additional arguments on top of `torch.nn.Linear`'s arguments:
        * `config`: Float8LinearConfig
        """

        config = kwargs.pop("config")
        super().__init__(*args, **kwargs)

        # Defines the scaling behavior of input, weight, grad_output
        self.scaling_type_input = config.cast_config_input.scaling_type
        self.scaling_type_weight = config.cast_config_weight.scaling_type
        self.scaling_type_grad_output = config.cast_config_grad_output.scaling_type
        self.config = config

        self.linear_mm_config = LinearMMConfig(
            # output
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_output.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_input
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_grad_input.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_weight
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_grad_weight.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Duplicate the autocast logic for F.linear, so that the output
        # of our module has the right original precision
        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            autocast_dtype = torch.get_autocast_gpu_dtype()
            input = input.to(autocast_dtype)

        output = matmul_with_hp_or_float8_args.apply(
            input,
            self.weight.t(),
            self.linear_mm_config,
            self.config,
        )

        return output

    def extra_repr(self):
        c = self.config
        ci = f"i:{c.cast_config_input.short_str()}"
        cw = f"w:{c.cast_config_weight.short_str()}"
        cgo = f"go:{c.cast_config_grad_output.short_str()}"
        parts = [ci, cw, cgo]
        if c.cast_config_input_for_grad_weight != c.cast_config_input:
            parts.append(f"i_gw:{c.cast_config_input_for_grad_weight.short_str()}")
        if c.cast_config_weight_for_grad_input != c.cast_config_weight:
            parts.append(f"w_gi:{c.cast_config_weight_for_grad_input.short_str()}")
        if c.cast_config_grad_output_for_grad_weight != c.cast_config_grad_output:
            parts.append(f"go_gw:{c.cast_config_grad_output_for_grad_weight.short_str()}")
        cast_config_str = ",".join(parts)
        s = f'{super().extra_repr()}, cast_configs={cast_config_str}"'
        return s

    @classmethod
    def from_float(
        cls,
        mod,
        config: Float8LinearConfig | None = None,
    ):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            config (Optional[Float8LinearConfig]): configuration for conversion to float8
        """
        if config is None:
            config = Float8LinearConfig()
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                config=config,
            )
        new_mod.weight = mod.weight

        # If FSDP float8 all-gather is on, wrap the weight in a float8-aware
        # tensor subclass. This must happen last because:
        # 1. weight needs to be on the correct device to create the buffers
        # 2. buffers need to be already created for the delayed scaling version
        #    of the weight wrapper to be initialized
        # TODO(future PR): see if we can simplify ^ now that delayed scaling is deleted
        if config.enable_fsdp_float8_all_gather:
            assert config.cast_config_weight.scaling_type is ScalingType.DYNAMIC
            new_mod.weight = torch.nn.Parameter(
                WeightWithDynamicFloat8CastTensor(
                    new_mod.weight,
                    new_mod.linear_mm_config,
                    new_mod.config.cast_config_weight.target_dtype,
                ),
                requires_grad=new_mod.weight.requires_grad,
            )

        return new_mod


def swap_linear_layers(
    module: nn.Module,
    from_float_func: Callable[[nn.Linear], nn.Linear],
    *,
    module_filter_fn: Callable[[nn.Module, str], bool] | None = None,
) -> nn.Module:
    """
    Generic function to swap linear layers in a module with a new type of linear layer.

    Note:
        If applied to a root-level nn.Linear, the module will not be modified in place
        and returned instead

    Args:
        module: Module to modify.
        from_float_func: Function that accepts a linear layer and returns a new type of linear layer.
        module_filter_fn: If specified, only the `torch.nn.Linear` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance, and the FQN.

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    if isinstance(module, Linear) and (module_filter_fn is None or module_filter_fn(module, "")):
        if len(list(module.children())) > 0:
            raise AssertionError(f"Does not support a root nn.Linear with children: {module}")
        return from_float_func(
            module,
        )

    root_module = module

    def post_order_traversal(
        module: nn.Module,
        cur_fqn: str | None = None,
        parent_module: nn.Module | None = None,
    ):
        if cur_fqn is None:
            cur_fqn = ""

        for child_module_name, child_module in module.named_children():
            if cur_fqn == "":
                new_fqn = child_module_name
            else:
                new_fqn = f"{cur_fqn}.{child_module_name}"

            post_order_traversal(child_module, new_fqn, module)

        if isinstance(module, Linear) and (module_filter_fn is None or module_filter_fn(module, cur_fqn)):
            assert parent_module is not None, f"Linear root module should return early: {module}"
            new_linear_module = from_float_func(module)
            cur_module_name = cur_fqn.split(".")[-1]
            setattr(parent_module, cur_module_name, new_linear_module)

    post_order_traversal(root_module)
    return root_module


def convert_to_float8_training(
    module: nn.Module,
    *,
    module_filter_fn=None,
    config: Float8LinearConfig | None = None,
) -> nn.Module:
    """
    Swaps `torch.nn.Linear` in `module` with `Float8Linear`.

    Args:
        module: Module to modify.
        module_filter_fn: If specified, only the `torch.nn.Linear` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance and the FQN.
        config (Float8LinearConfig): configuration for conversion to float8

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    torch._C._log_api_usage_once("torchao.float8.convert_to_float8_training")
    if config is None:
        config = Float8LinearConfig()

    from_float = lambda m: Float8Linear.from_float(
        m,
        config=config,
    )

    return swap_linear_layers(
        module,
        from_float,
        module_filter_fn=module_filter_fn,
    )


torch.cuda.set_device("cuda:2")

if __name__ == "__main__":
    train_path = "/mnt/harddrive/datasets/dclm-edu/tokenized_superbpe"
    validation_path = "data_dclm_edu/tokenized_superbpe_large/val.npy"
    override = {
        "model.vocab_size": 32768,
        "data.batch_size": 96,
        "data.context_length": 640,
        "model.attn_qknorm": True,
        "trainer.max_steps": 10,
        "model.attn_gating": "per-head",
        "trainer.save_every": 2500,
        "optim.warmup_steps": 1000,
        "optim.muon_lr": 2.2e-2,
        "optim.muon_wd": 0.1,
        "data.tokenizer_path": "data_dclm_edu/tokenizer_superbpe_hf",
        "trainer.gradient_accumulation_steps": 1,
        "trainer.dist_mode": "zero2",
    }
    override = {
        **override,
        "data.train_path": train_path,
        "data.validation_path": validation_path,
    }

    cfg = apply_overrides(cfg, override)
    overrides = {"optim.cosine_steps": cfg.trainer.max_steps}
    cfg = apply_overrides(cfg, overrides)
    run = None
    trainer = Trainer(cfg, wandb=run, compile=False)

    # optional: filter modules from being eligible for float8 conversion
    def module_filter_fn(mod: torch.nn.Module, fqn: str):
        # don't convert the last module
        if fqn == "1":
            return False
        # don't convert linear modules with weight dimensions not divisible by 16
        if isinstance(mod, Linear):
            if mod.in_features % 8 != 0 or mod.out_features % 8 != 0:
                return False
        return True

    # convert specified `torch.nn.Linear` modules to `Float8Linear`
    convert_to_float8_training(trainer.model, module_filter_fn=module_filter_fn)

    print(trainer.model)

    # enable torch.compile for competitive performance
    trainer.model = torch.compile(trainer.model)

    # warmup
    trainer.train()

    t0 = time.monotonic()
    # toy training loop
    trainer.train()

    print(time.monotonic() - t0)
