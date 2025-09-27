from dataclasses import dataclass
from typing import Literal
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    train_path: str | Path = ""
    validation_path: str | Path = ""
    batch_size: int = 1
    val_batch_size: int = 1
    context_length: int = 1024
    seed: int = 42


@dataclass(frozen=False)
class OptimConfig:
    lr: float = 7e-3
    wd: float = 1e-7
    betas: tuple[float, float] = (0.9, 0.99)
    lr_min_coeff: float = 1e-2
    scheduler: str = "cosine"  # or "wsd"
    use_muon: bool = False
    use_lion: bool = False
    # if None then use lr / wd
    muon_lr: float | None = None
    muon_wd: float | None = None
    muon_wd_min: float | None = None
    warmup_steps: int = 1000
    cosine_steps: int = 10_000
    stable_steps: int = 10_000
    decay_steps: int = 10_000


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 1024
    d_ff: int = 4096
    n_layers: int = 24
    n_heads: int = 16
    vocab_size: int = 10_000
    theta: float = 10_000
    weight_tying: bool = False
    attn_qknorm: bool = False
    attn_val_residual: bool = False
    attn_gating: bool = False
    layernorm_scaling: bool = False


@dataclass(frozen=False)
class TrainerConfig:
    load_from: str | None = None
    device: str = "cuda:2"
    dtype: Literal["float32", "bfloat16"] = "float32"
    max_steps: int = 200_000
    z_loss_weight: float = 1e-4
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    # run_name: str = "{date}_{optim.lr}"  # template
    run_name: str = "{date}"  # template
    save_dir: str | Path = Path(__file__).parent / "checkpoints"
    # save every n steps
    save_every: int = 100
    # validate every n steps
    val_every: int = 100
    # log train metrics every n steps
    log_every: int = 10


@dataclass(frozen=False)
class Config:
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    trainer: TrainerConfig
    project: str = "cs336"  # for wandb


default_cfg = Config(DataConfig(), ModelConfig(), OptimConfig(), TrainerConfig())
