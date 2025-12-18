# SmolLM2-135M → IMU1: conversion + GQA + widening experiments

This file is meant to be a reproducible record of *actually executed* conversions/evals in this workspace (with exact checkpoint paths and scores).

## Environment

- Python: `/home/george/axolotl/.venv/bin/python`
- Launcher: `/home/george/axolotl/.venv/bin/torchrun --nproc_per_node 4`
- Eval: `sample_efficient_gpt/evals/base_eval.py`
- Tokenizer: `--tokenizer_path HuggingFaceTB/SmolLM2-135M`

## Notes / assumptions

- SmolLM2-135M is LLaMA-shaped and uses GQA (`num_key_value_heads=3`, `num_attention_heads=9`, `head_dim=64`).
- `sample_efficient_gpt` now supports native GQA via `model.n_kv_heads` (no KV head repetition in the converter).
- For widening, `--n-kv-heads auto` keeps the HF head-group-size constant: `group = n_heads / n_kv_heads` (SmolLM2: `9/3 = 3`), so widening to 18 heads uses 6 KV heads.

## Converter

Script: `sample_efficient_gpt/scripts/hf/convert_smollm2_to_imu.py`

Widening modes:
- `noise`: keep muP init in expanded regions (scaled by `--init-multiplier`)
- `preserve`: zero-fill expanded regions and rescale RMSNorm gains (function-preserving widening)
- `preserve-norm`: keep muP init in expanded regions but rescale RMSNorm gains (best “widen without collapsing” here)

## Experiments

### 1) Baseline conversion (GQA, no widening)

Convert:
```bash
/home/george/axolotl/.venv/bin/python sample_efficient_gpt/scripts/hf/convert_smollm2_to_imu.py \
  --hf-dir HuggingFaceTB/SmolLM2-135M \
  --output /tmp/smollm2_imu_baseline_gqa.pt \
  --width from-hf --n-heads from-hf --n-kv-heads from-hf --d-ff from-hf \
  --n-layers from-hf --vocab-size from-hf --theta from-hf \
  --init-multiplier 1.0 --widening-mode noise
```

Eval (`arc_easy`, full):
```bash
/home/george/axolotl/.venv/bin/torchrun --nproc_per_node 4 sample_efficient_gpt/evals/base_eval.py \
  --checkpoint /tmp/smollm2_imu_baseline_gqa.pt --type se --task arc_easy \
  --tokenizer_path HuggingFaceTB/SmolLM2-135M --batch 8
```

Result:
- `arc_easy` accuracy: `0.645202` (centered `0.526936`)

### 2) 2× widening baseline (1152 width, 18 heads, 6 KV heads, 3072 FFN)

Convert:
```bash
/home/george/axolotl/.venv/bin/python sample_efficient_gpt/scripts/hf/convert_smollm2_to_imu.py \
  --hf-dir HuggingFaceTB/SmolLM2-135M \
  --output /tmp/smollm2_imu_wide2x_gqa_preservenorm_m1e-5.pt \
  --width 1152 --n-heads 18 --n-kv-heads auto --d-ff 3072 \
  --n-layers from-hf --vocab-size from-hf --theta from-hf \
  --init-multiplier 1e-5 --widening-mode preserve-norm
```

Eval (`arc_easy`, full):
```bash
/home/george/axolotl/.venv/bin/torchrun --nproc_per_node 4 sample_efficient_gpt/evals/base_eval.py \
  --checkpoint /tmp/smollm2_imu_wide2x_gqa_preservenorm_m1e-5.pt --type se --task arc_easy \
  --tokenizer_path HuggingFaceTB/SmolLM2-135M --batch 8
```

Result:
- `arc_easy` accuracy: `0.647306` (centered `0.529742`)

### 3) 2× widening + extras: `attn_gating=per-head`, `attn_val_residual=true`, `attn_qknorm=true`

Implementation notes:
- Value residual is initialized as a no-op (`alpha1=1`, `alpha2=0`, `scale=1`).
- Per-head gating uses `2*sigmoid(gate)`, and the converter zeros the gate weights so the initial behavior is neutral.
- QK-norm implementation uses the same Triton attention kernel as the baseline path 

Base convert (gain initialized to 8.0):
```bash
/home/george/axolotl/.venv/bin/python sample_efficient_gpt/scripts/hf/convert_smollm2_to_imu.py \
  --hf-dir HuggingFaceTB/SmolLM2-135M \
  --output /tmp/smollm2_imu_wide2x_gqa_extras_qknormrms_g1.pt \
  --width 1152 --n-heads 18 --n-kv-heads auto --d-ff 3072 \
  --n-layers from-hf --vocab-size from-hf --theta from-hf \
  --init-multiplier 1e-5 --widening-mode preserve-norm \
  --attn-gating per-head --attn-val-residual --attn-qknorm --qknorm-gain 8.0
```

Gain sweep (quick, `--max_per_task 200`):
- `gain=0.125`: `0.270000`
- `gain=0.25`: `0.305000`
- `gain=0.5`: `0.315000`
- `gain=1.0`: `0.330000`
- `gain=2.0`: `0.330000`
- `gain=4.0`: `0.345000`
- `gain=8.0`: `0.375000` (best quick)
- `gain=16.0`: `0.225000`

Full eval for best quick gain:
```bash
/home/george/axolotl/.venv/bin/torchrun --nproc_per_node 4 sample_efficient_gpt/evals/base_eval.py \
  --checkpoint /tmp/smollm2_imu_wide2x_gqa_extras_qknormrms_g8.0.pt --type se --task arc_easy \
  --tokenizer_path HuggingFaceTB/SmolLM2-135M --batch 8
```

Result (full):
- `arc_easy` accuracy: `0.322811` (centered `0.097082`)
