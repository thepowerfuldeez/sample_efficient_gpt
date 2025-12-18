## Sample Efficient GPT

This repo is a training framework that is driving my sample efficiency experiments for pre-training LLMs.

A goal to explore the frontier of sample efficiency of small language models.

Main achievements that makes this framework stand out across others:

1. Gated Attention (https://arxiv.org/abs/2505.06708)
2. Value Residual Learning (https://arxiv.org/abs/2410.17897)
3. Muon + Triton implementation from Dion repo
4. LayerNorm Scaling (https://arxiv.org/abs/2502.05795)
5. QK-Norm + I wrote Flash attention QK-norm kernel for max efficiency
6. Z-loss (https://arxiv.org/pdf/2204.02311)
7. muP parametrization (reference from https://arxiv.org/abs/2505.02222)
8. SuperBPE tokenizer with the conversion to HF tokenizers (https://arxiv.org/abs/2503.13423)
9. Individual WD for muP transfer + Cautious Weight Decay (https://arxiv.org/abs/2510.12402)

Experimental (use at your own risk):
10. Token-level MoE FFN (top-k routing, load-balancing aux loss)


Various optimization tricks, such as momentum warmup, WD schedule



### Quickstart

```bash
uv sync
```

1. Train tokenizer and tokenize data (skip if you already have tokenizer)

- Train a tokenizer:

    (Look into tokenizer/README.md -- TBD)

- Tokenize data
    ```bash
    uv run tokenize_with_fast_tokenizer.py --data-path DATA_PATH --tokenized-data-path TOKENIZED_DATA_PATH --include-val-data=1 --tokenizer-name HF_TOKENIZER_NAME
    ```

    `DATA_PATH` is expected to be a folder with either parquet or txt files

    In case of txt files, `TOKENIZED_DATA_PATH` will contain .npy files with matching names with tokenized data

    In case of parquet, `TOKENIZED_DATA_PATH` will contain .npz files for every chunk in your dataset

    `HF_TOKENIZER_NAME` is a huggingface transformers format, can accept hf name or folder

2. Run training

Sample script to run 70M LLM training on RTX 5090 (note that I specify device "cuda:2" in a script):
```bash
bash baseline.sh
```

You can modify k/v pairs in config with --override argument to `train.py`

Note that you have to provide both train and validation datasets in tokenized format as npy files.

### To-do:
* DDP (X)
* FSDP
* Evals (X)
* FP8 training (+-)
* MoE (experimental ✅)

Not fully verified:
- Seesaw schedule
- Prodigy optimizer

### Quick MoE try (B200-friendly)
Set these in your YAML (e.g. `sample_efficient_gpt/configs/smollm2_wide.yaml` under `model:`) to increase parameter count at ~same per-token compute:
```yaml
model:
  moe_num_experts: 8        # 0 disables MoE
  moe_top_k: 1              # top-1 routing (Switch-style)
  moe_capacity_factor: 1.25 # avoid drops; increase if you see drops/instability
  moe_aux_loss_coef: 0.01   # load balancing
  moe_z_loss_coef: 0.0      # optional router z-loss
  moe_router_jitter: 0.0    # e.g. 0.01 for noisy routing
  moe_expert_parallel_size: 1 # set == world_size to shard experts (top_k must be 1)
  moe_expert_precision: bf16  # bf16 (default) or fp8 (experts only; requires torchao)
  moe_start_layer: 0
  moe_every_n_layers: 1     # apply MoE to every layer
  moe_end_layer: null
trainer:
  compile: false            # MoE routing is Python-heavy; disable torch.compile initially
```

Useful perf toggles (environment variables):
- `SEGPT_RMSNORM_IMPL=liger` (fused RMSNorm)
- `SEGPT_SWIGLU_IMPL=liger` (fused SiLU*mul inside SwiGLU)
- `SEGPT_ATTN_IMPL=triton` / `SEGPT_QKNORM_ATTN_IMPL=triton` (Triton attention kernels)
- `SEGPT_TRACK_KURTOSIS=0` (disable kurtosis computation)

### B200 quickstart
- `docs/B200_QUICKSTART.md`
