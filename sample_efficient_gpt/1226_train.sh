export SEGPT_RMSNORM_IMPL=liger
export SEGPT_SWIGLU_IMPL=liger
export SEGPT_TRACK_KURTOSIS=0
export SEGPT_SONICMOE_ACT_SCALE=0.5
export SEGPT_SONICMOE_ACT_CLAMP=20.0

uv run torchrun --nproc_per_node 8 sample_efficient_gpt/train.py \
  --config sample_efficient_gpt/configs/moe_runs.yaml \
  --config-key smollm2_moe49 \
  --world-size 8