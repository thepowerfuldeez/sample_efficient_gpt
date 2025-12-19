from dataclasses import dataclass
from typing import Literal
from pathlib import Path


@dataclass(frozen=False)
class DataConfig:
    train_path: str | Path = ""
    validation_path: str | Path = ""
    tokenizer_path: str | Path = ""
    batch_size: int = 1
    val_batch_size: int = 1
    context_length: int = 1024
    seed: int = 42
    sample_packing: bool = False
    pad_token_id: int = 0
    mode: str = "pretrain"  # "pretrain" or "sft"


@dataclass(frozen=False)
class OptimConfig:
    lr: float = 7e-3
    wd: float = 1e-7
    betas: tuple[float, float] = (0.9, 0.99)
    lr_min_coeff: float = 1e-2
    scheduler: str = "cosine"  # or "wsd" or "seesaw"
    use_muon: bool = False
    use_lion: bool = False
    # if None then use lr / wd
    muon_lr: float | None = None
    muon_wd: float | None = None
    muon_wd_min: float | None = None
    # for model surgery, run with 0 lr for some time
    zero_lr_steps: int = 0
    warmup_steps: int = 1000
    cosine_steps: int = 10_000
    seesaw_steps: tuple[int] = tuple()
    wsd_need_warmup: bool = True
    wsd_phase: str = "stable"  # stable or decay
    wsd_decay_step: str | None = None  # current step if None


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 1024
    d_ff: int = 4096
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int | None = None
    vocab_size: int = 10_000
    theta: float = 10_000
    rope_interleaved: bool = False
    weight_tying: bool = False
    attn_qknorm: bool = False
    attn_val_residual: bool = False
    attn_gating: bool = False
    layernorm_scaling: bool = False
    # --- MoE (token-level, no expert-parallel) ---
    # MoE implementation backend.
    # - "native": this repo's TopKMoE (supports EP, fp8 experts via torchao)
    # - "sonicmoe": Dao-AILab/sonic-moe (single-rank experts only; requires external install)
    moe_backend: str = "native"
    # 0 disables MoE
    moe_num_experts: int = 0
    moe_top_k: int = 1
    moe_capacity_factor: float = 1.0
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 0.0
    moe_router_jitter: float = 0.0
    moe_normalize_gates: bool = True
    # Scales router gates applied to expert outputs (e.g. 10.0 increases router/expert signal strength).
    moe_gate_scale: float = 1.0
    # expert parallelism (currently supports moe_expert_parallel_size == world_size)
    moe_expert_parallel_size: int = 1
    # expert matmul precision mode (best-effort; may fall back depending on environment)
    moe_expert_precision: str = "bf16"  # "bf16" or "fp8"
    # apply MoE to layers in [start, end) every N layers
    moe_start_layer: int = 0
    moe_every_n_layers: int = 1
    moe_end_layer: int | None = None


@dataclass(frozen=False)
class TrainerConfig:
    load_from: str | None = None
    device: str = "cuda"
    dtype: Literal["float32", "bfloat16"] = "float32"
    use_fp8: bool = False
    compile: bool = True
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
    # distributed training mode (currently only 'ddp' is supported)
    dist_mode: str = "ddp"


@dataclass(frozen=False)
class Config:
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    trainer: TrainerConfig
    project: str = "cs336"  # for wandb


default_cfg = Config(DataConfig(), ModelConfig(), OptimConfig(), TrainerConfig())
