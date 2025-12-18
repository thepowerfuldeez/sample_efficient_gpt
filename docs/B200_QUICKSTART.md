# B200 Quickstart (SmolLM2-shape MoE-32, EP+FP8 experts)

This uses:
- Converter: `sample_efficient_gpt/scripts/hf/convert_smollm2_to_imu.py`
- Training config: `sample_efficient_gpt/configs/moe_runs.yaml` (`smollm2_moe32_ep_fp8_b200_8gpu`)

## Host setup

```bash
sudo apt install zstd wget htop tmux git python3-dev -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## Repo setup

```bash
uv sync
export UV_TORCH_BACKEND=auto
uv pip install setuptools uv_build maturin
uv sync
uv run wandb login
uv tool install hf
uv run hf auth
```

## Data (example)

```bash
hf download thepowerfuldeez/imu1_base_stable_corpus --repo-type=dataset
uv run tokenizer/split_val.py \
  --tokenized-data-path "/root/.cache/huggingface/hub/datasets--thepowerfuldeez--imu1_base_stable_corpus/snapshots/3dfa2f3a75603cca4de91ad8a61407e4225358ce/data/dclm_edu" \
  --save-dir /root/
```

This produces `/root/train.npy` and `/root/val.npy` (adjust paths if your split script differs).

## Convert SmolLM2 → IMU1-MoE (save to `/root`)

This keeps the original SmolLM2 shape (`d_model=576,d_ff=1536,n_layers=30`) and “moefies” the FFNs.

```bash
uv run sample_efficient_gpt/scripts/hf/convert_smollm2_to_imu.py \
  --hf-dir HuggingFaceTB/SmolLM2-135M \
  --output /root/smollm2_imu_moe32.pt \
  --width from-hf --n-heads from-hf --n-kv-heads from-hf --d-ff from-hf \
  --n-layers from-hf --vocab-size from-hf --theta from-hf \
  --attn-qknorm --attn-val-residual --attn-gating per-head --layernorm-scaling --weight-tying \
  --moe-num-experts 32 --moe-top-k 1 --moe-capacity-factor 0.0 \
  --moe-start-layer 0 --moe-end-layer 30 --moe-every-n-layers 1 \
  --moe-expert-init permute --moe-permute-seed 0 --moe-router-bias-init 10.0
```

## Train (8×B200, EP+FP8 experts)

```bash
export SEGPT_RMSNORM_IMPL=liger
export SEGPT_SWIGLU_IMPL=liger
export SEGPT_TRACK_KURTOSIS=0

uv run torchrun --nproc_per_node 8 sample_efficient_gpt/train.py \
  --config sample_efficient_gpt/configs/moe_runs.yaml \
  --config-key smollm2_moe32_ep_fp8_b200_8gpu \
  --world-size 8 \
  --train-path /root/train.npy \
  --validation-path /root/val.npy \
  --load-from /root/smollm2_imu_moe32.pt
```

Notes:
- EP currently requires `model.moe_expert_parallel_size == world_size` and `model.moe_top_k == 1`.
- EP checkpoints are saved per-rank as `*.rank{rank}.pt`; optimizer resume for EP is not implemented yet.

