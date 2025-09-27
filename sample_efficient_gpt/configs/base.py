from sample_efficient_gpt.config_schema import Config, DataConfig, ModelConfig, OptimConfig, TrainerConfig

cfg = Config(
    data=DataConfig(),
    model=ModelConfig(),
    optim=OptimConfig(),
    trainer=TrainerConfig(),
)
