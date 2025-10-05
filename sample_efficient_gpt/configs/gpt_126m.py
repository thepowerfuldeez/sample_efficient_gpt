from sample_efficient_gpt.config_schema import Config, ModelConfig, DataConfig, TrainerConfig, OptimConfig
from sample_efficient_gpt.configs.base import cfg as base


cfg = Config(
    data=DataConfig(
        base.data.train_path,
        base.data.validation_path,
        batch_size=24,
        val_batch_size=48,
        context_length=512,
        seed=42,
    ),
    # layer params = 7.3M, total non-emb=24.5M + 7.3*12 = 112.5M
    model=ModelConfig(
        vocab_size=32000,
        attn_qknorm=True,
        attn_val_residual=True,
        layernorm_scaling=True,
        d_model=1024,
        d_ff=2304,
        n_layers=12,
        n_heads=16,
        n_kv_heads=8,
    ),
    # wd is halved, lr is scaled to 768/1024
    optim=OptimConfig(
        lr=5.25e-3,
        lr_min_coeff=5e-2,
        wd=0.0,
        use_muon=True,
        betas=(0.95, 0.99),
        muon_lr=1.125e-2,
        muon_wd=1.5e-3,
        muon_wd_min=1.5e-4,
    ),
    # kept global batch size the same: 192
    trainer=TrainerConfig(
        log_every=50, save_every=4000, val_every=3000, max_steps=24000, gradient_accumulation_steps=8, dtype="bfloat16"
    ),
    project=base.project,
)
