from sample_efficient_gpt.config_schema import Config, ModelConfig, DataConfig, TrainerConfig, OptimConfig
from sample_efficient_gpt.configs.base import cfg as base


cfg = Config(
    data=DataConfig(
        base.data.train_path,
        base.data.validation_path,
        batch_size=96,
        val_batch_size=192,
        context_length=384,
        seed=42,
    ),
    # layer params = 5.5M, total non-emb=24.5M + 5.5*8 = 68.5M
    model=ModelConfig(
        vocab_size=32000,
        attn_qknorm=True,
        attn_val_residual=True,
        layernorm_scaling=True,
        d_model=768,
        d_ff=2048,
        n_layers=8,
        n_heads=12,
    ),
    optim=OptimConfig(
        lr=7e-3, lr_min_coeff=1e-2, wd=0.0, use_muon=True, betas=(0.95, 0.99), muon_lr=1.5e-2, muon_wd=1e-4
    ),
    trainer=TrainerConfig(
        log_every=50, save_every=4000, val_every=3000, max_steps=24000, gradient_accumulation_steps=2, dtype="bfloat16"
    ),
    project=base.project,
)
