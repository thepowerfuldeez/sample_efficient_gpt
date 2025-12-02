sudo apt update
sudo apt install zstd wget htop tmux git python3-dev -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
export UV_TORCH_BACKEND=auto
uv pip install setuptools uv_build maturin
uv tool install hf
uv run hf auth
uv sync
uv run wandb login

hf download thepowerfuldeez/imu_1_base
hf download thepowerfuldeez/imu1_base_stable_corpus --repo-type=dataset
hf download thepowerfuldeez/imu1_base_decay_corpus --repo-type=dataset
uv run tokenizer/split_val.py --tokenized-data-path "/root/.cache/huggingface/hub/datasets--thepowerfuldeez--imu1_base_stable_corpus/snapshots/3dfa2f3a75603cca4de91ad8a61407e4225358ce/data/dclm_edu" --save-dir /root/

uv run torchrun --nproc_per_node 8 train.py --config configs/main_runs.yaml --config-key main_run_mix_stable_continue --world-size 8