# ensure that fastsplit and rustbpe are installed
uv tool install maturin
cd ../fastsplit && maturin develop --release
cd ../rustbpe && maturin develop --release
cd ../tokenizer

# train
uv run train_bpe.py --vocab-size 65536 --data-path ../data_mix/train.txt --save-dir trained_tokenizers/mix_bpe --use-rust-bpe 1
# convert to hf
uv run convert_tokenizer_to_hf.py --tokenizer-path trained_tokenizers/mix_bpe --save-path trained_tokenizers/mix_bpe_hf


# tokenize
# uv run tokenize_with_fast_tokenizer.py --tokenizer-name trained_tokenizers/dclm_edu_bpe_hf/ --data-path /mnt/harddrive/datasets/dclm-edu/data/ --tokenized-data-path /mnt/harddrive/datasets/dclm-edu/tokenized_65536
# split val
# uv run split_val.py --tokenized-data-path /mnt/harddrive/datasets/dclm-edu/tokenized_65536 --save-dir /home/george/sample_efficient_gpt/sample_efficient_gpt/data_dclm_edu/tokenized_65536