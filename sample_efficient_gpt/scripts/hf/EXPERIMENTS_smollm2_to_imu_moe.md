# SmolLM2-135M → IMU1-MoE: upcycling dense FFN into MoE experts

This file mirrors `sample_efficient_gpt/scripts/hf/EXPERIMENTS_smollm2_to_imu.md`, but for the MoE (upcycled FFN) path.

## Converter

Script: `sample_efficient_gpt/scripts/hf/convert_smollm2_to_imu.py`

Key flags (new):
- `--moe-num-experts N`
- `--moe-start-layer L0`, `--moe-every-n-layers N`, `--moe-end-layer L1`
- `--moe-expert-init permute` (function-preserving hidden permutation per expert)
- `--moe-capacity-factor 0.0` (disable capacity drops; helps function-preserving conversion)
- `--moe-router-bias-init 10.0` (route almost everything to expert0; keeps gates ~1)

### 1) Baseline IMU1-MoE conversion (no widening)

Convert (top-1, route-to-expert0 init via zero router):
```bash
/home/george/axolotl/.venv/bin/python sample_efficient_gpt/scripts/hf/convert_smollm2_to_imu.py \
  --hf-dir HuggingFaceTB/SmolLM2-135M \
  --output /tmp/smollm2_imu1_moe_e8.pt \
  --width from-hf --n-heads from-hf --n-kv-heads from-hf --d-ff from-hf \
  --n-layers from-hf --vocab-size from-hf --theta from-hf \
  --init-multiplier 1.0 --widening-mode noise \
  --attn-qknorm --attn-val-residual --attn-gating per-head --layernorm-scaling --weight-tying \
  --moe-num-experts 32 --moe-top-k 1 --moe-capacity-factor 0.0 \
  --moe-start-layer 0 --moe-every-n-layers 1 --moe-end-layer 30 --moe-router-bias-init 10.0 \
  --moe-expert-init permute --moe-permute-seed 0
```

Eval (`arc_easy`, full):
```bash
/home/george/axolotl/.venv/bin/torchrun --nproc_per_node 4 sample_efficient_gpt/evals/base_eval.py \
  --checkpoint /tmp/smollm2_imu1_moe_e8.pt --type se --task arc_easy \
  --tokenizer_path HuggingFaceTB/SmolLM2-135M --batch 8
```

Result:
- Dense (features enabled, no MoE): `0.276515` (centered `0.035354`) using `/tmp/smollm2_imu_dense_features.pt`
- MoE-32 (features enabled, no-drop init): `0.279040` (centered `0.038721`) using `/tmp/smollm2_imu_moe32_features_nodrop.pt`
