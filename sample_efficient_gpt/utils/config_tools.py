import json
from dataclasses import asdict, replace
from typing import Any
from datetime import datetime

from sample_efficient_gpt.config_schema import (
    Config,
    default_cfg,
    DataConfig,
    ModelConfig,
    TrainerConfig,
    OptimConfig,
)


def dataclass_to_nested_dict(dc) -> dict[str, Any]:
    return asdict(dc)


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """
    Flatten the dict
    """
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def set_by_dotted(dc, dotted_key: str, value: Any):
    parts = dotted_key.split(".")

    def _set(obj, idx=0):
        if idx == len(parts) - 1:
            return replace(obj, **{parts[idx]: value})
        child = getattr(obj, parts[idx])
        new_child = _set(child, idx + 1)
        return replace(obj, **{parts[idx]: new_child})

    return _set(dc)


def apply_overrides(cfg, overrides: dict[str, Any]):
    for k, v in overrides.items():
        cfg = set_by_dotted(cfg, k, v)
    return cfg


def save_config(cfg: Config):
    return json.dumps(dataclass_to_nested_dict(cfg), default=str)


def load_config(config_str: str):
    cfg: Config = default_cfg
    updated_cfg = apply_overrides(cfg, json.loads(config_str))
    updated_cfg = Config(
        data=DataConfig(**updated_cfg.data),
        model=ModelConfig(**updated_cfg.model),
        trainer=TrainerConfig(**updated_cfg.trainer),
        optim=OptimConfig(**updated_cfg.optim),
    )
    return updated_cfg


def render_template(template: str, cfg, extra: dict[str, Any] | None = None) -> str:
    base = flatten(dataclass_to_nested_dict(cfg))
    base["date"] = datetime.now().strftime("%m%d")
    base["time"] = datetime.now().strftime("%H%M")
    if extra:
        base.update(extra)

    class DotDict(dict):
        def __missing__(self, key):
            return "NA"

    return template.format_map(base)


def wandb_run_name(cfg: Config, extra: dict[str, Any] | None = None) -> str:
    return render_template(cfg.trainer.run_name, cfg, extra)
