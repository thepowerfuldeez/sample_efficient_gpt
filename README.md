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
* DDP
* FSDP
* Evals
* FP8 training
* MoE
