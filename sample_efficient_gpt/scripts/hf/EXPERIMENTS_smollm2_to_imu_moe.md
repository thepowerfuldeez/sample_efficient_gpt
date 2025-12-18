# SmolLM2-135M → IMU1-MoE: upcycling dense FFN into MoE experts

This file mirrors `sample_efficient_gpt/scripts/hf/EXPERIMENTS_smollm2_to_imu.md`, but for the MoE (upcycled FFN) path.

## Converter

Script: `sample_efficient_gpt/scripts/hf/convert_smollm2_to_imu.py`

Key flags (new):
- `--moe-num-experts N`
- `--moe-start-layer L0`, `--moe-every-n-layers N`, `--moe-end-layer L1`
- `--moe-expert-init permute` (function-preserving hidden permutation per expert)

### 1) Baseline IMU1-MoE conversion (no widening)

Convert (top-1, route-to-expert0 init via zero router):
```bash
/home/george/axolotl/.venv/bin/python sample_efficient_gpt/scripts/hf/convert_smollm2_to_imu.py \
  --hf-dir HuggingFaceTB/SmolLM2-135M \
  --output /tmp/smollm2_imu1_moe_e8.pt \
  --width from-hf --n-heads from-hf --n-kv-heads from-hf --d-ff from-hf \
  --n-layers from-hf --vocab-size from-hf --theta from-hf \
  --init-multiplier 1.0 --widening-mode noise \
  --moe-num-experts 8 --moe-top-k 1 \
  --moe-start-layer 0 --moe-every-n-layers 1 --moe-end-layer 30 \
  --moe-expert-init permute --moe-permute-seed 0
```

Eval (`arc_easy`, full):
```bash
/home/george/axolotl/.venv/bin/torchrun --nproc_per_node 4 sample_efficient_gpt/evals/base_eval.py \
  --checkpoint /tmp/smollm2_imu1_moe_e8.pt --type se --task arc_easy \
  --tokenizer_path HuggingFaceTB/SmolLM2-135M --batch 8
```

Result:
- `arc_easy` accuracy: `TBD`

