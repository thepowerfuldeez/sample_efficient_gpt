# SonicMoE backend (optional)

This repo can optionally use Dao-AILab/sonic-moe as the MoE FFN backend (instead of the native `TopKMoE`).

## 1) Get the code

This repo tracks SonicMoE as a git submodule:

```bash
git submodule update --init --recursive
```

## 2) Install dependencies

SonicMoE currently depends on extra packages (e.g. `nvidia-cutlass-dsl` and `quack-kernels`). Install per the upstream README:

- `ext/sonic-moe/README.md`
- `ext/sonic-moe/requirements.txt`

One common install sequence:

```bash
uv pip install -r ext/sonic-moe/requirements.txt
uv pip install -e ext/sonic-moe
```

## 3) Enable in config

SonicMoE is enabled via `model.moe_backend: sonicmoe`. The current integration is intentionally conservative:

- No expert-parallel (set `moe_expert_parallel_size: 1`)
- No expert fp8 conversion via torchao (set `moe_expert_precision: bf16`)
- No capacity/drop logic (set `moe_capacity_factor: 1.0`)
- No z-loss (set `moe_z_loss_coef: 0.0`)
- No extra gate scaling (set `moe_gate_scale: 1.0`)

Example:

```yaml
model:
  moe_backend: sonicmoe
  moe_num_experts: 128
  moe_top_k: 8
  moe_aux_loss_coef: 0.01
  moe_expert_parallel_size: 1
  moe_expert_precision: bf16
  moe_capacity_factor: 1.0
  moe_z_loss_coef: 0.0
  moe_gate_scale: 1.0
trainer:
  compile: false
```

## Notes

- SonicMoE is designed for Hopper-class GPUs; correctness may still work elsewhere, but performance (and even kernel availability) may vary.
- This backend is not checkpoint-compatible with the native MoE implementation (different parameter layouts).

