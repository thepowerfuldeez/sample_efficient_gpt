# Imu1 inference (native, HF, vLLM)

This note captures the working commands and expected inputs for running the Imu1 checkpoint in three formats.

## Native (sample_efficient_gpt)

Use the native sampler on the original `.pt` checkpoint.

```bash
source /home/george/sample_efficient_gpt/.venv/bin/activate
python /home/george/sample_efficient_gpt/sample_efficient_gpt/utils/sample_outputs.py \
  --checkpoint /mnt/harddrive/checkpoints/0101_smollm_imu_midtrain/265000.pt \
  --prompt "The capital of France is" \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --top-p 0.95 \
  --max-steps 16
```

## HF conversion + HF inference

Convert the checkpoint into HF format with a tokenizer directory that supplies BOS/EOS/PAD ids. The local HF cache works for SmolLM2:

```bash
source /home/george/sample_efficient_gpt/.venv/bin/activate
python /home/george/sample_efficient_gpt/sample_efficient_gpt/scripts/hf/convert_imu1_checkpoint.py \
  --checkpoint /mnt/harddrive/checkpoints/0101_smollm_imu_midtrain/265000.pt \
  --output-dir /home/george/sample_efficient_gpt/imu1_converted_265000_reconv \
  --tokenizer-dir /home/george/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --test
```

Run HF generation against the converted output:

```bash
source /home/george/sample_efficient_gpt/.venv/bin/activate
python /home/george/sample_efficient_gpt/sample_efficient_gpt/scripts/hf/run_imu1_hf.py \
  --model-dir /home/george/sample_efficient_gpt/imu1_converted_265000_reconv \
  --tokenizer /home/george/sample_efficient_gpt/imu1_converted_265000_reconv \
  --prompt "The capital of France is" \
  --max-new-tokens 16 \
  --top-p 0.95 \
  --temperature 0.7
```

## vLLM (infer_vllm_k.py)

Use the vLLM venv and ensure vLLM can import its source and CUDA runtime library:

```bash
source /home/george/vllm/.venv/bin/activate
export PYTHONPATH=/home/george/vllm
export LD_LIBRARY_PATH=/home/george/learning/.venv/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

Run the inference script with a small smoke test (adjust `--limit`, `--k`, and `--max-new-tokens` as needed):

```bash
torchrun --standalone --nproc_per_node=1 \
  /home/george/sample_efficient_gpt/sample_efficient_gpt/scripts/data_processing/sft/infer_vllm_k.py \
  --model-dir /home/george/sample_efficient_gpt/imu1_converted_265000_reconv \
  --data-file /home/george/sample_efficient_gpt/sample_efficient_gpt/evals/train0_filtered.jsonl \
  --limit 1 \
  --k 1 \
  --max-new-tokens 16 \
  --output-dir /home/george/sample_efficient_gpt/_vllm_smoke \
  --prefix imu1_smoke.jsonl
```

Output JSONL files are written to `--output-dir` and include `generated_outputs` per prompt. The first run will capture CUDA graphs, which can take ~1 minute on this host.
