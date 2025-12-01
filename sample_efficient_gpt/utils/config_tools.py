import importlib
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from sample_efficient_gpt.config_schema import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TrainerConfig,
    default_cfg,
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


def _merge_section(section: dict[str, Any] | None, defaults) -> dict[str, Any]:
    merged = dataclass_to_nested_dict(defaults)
    if section:
        merged.update(section)
    return merged


def _load_base_cfg(base_name: str | None) -> Config:
    if not base_name:
        return default_cfg
    module_name = base_name
    if not module_name.startswith("sample_efficient_gpt.configs"):
        module_name = f"sample_efficient_gpt.configs.{module_name}"
    module = importlib.import_module(module_name)
    cfg = getattr(module, "cfg", None)
    if cfg is None:
        raise ValueError(f"Config module {module_name} does not define cfg")
    return cfg


def config_from_dict(config: dict[str, Any], base_cfg: Config = default_cfg) -> Config:
    cfg = apply_overrides(base_cfg, flatten(config))
    return Config(
        data=DataConfig(**dataclass_to_nested_dict(cfg)["data"]),
        model=ModelConfig(**dataclass_to_nested_dict(cfg)["model"]),
        trainer=TrainerConfig(**dataclass_to_nested_dict(cfg)["trainer"]),
        optim=OptimConfig(**dataclass_to_nested_dict(cfg)["optim"]),
        project=config.get("project", cfg.project),
    )


def save_config(cfg: Config):
    return json.dumps(dataclass_to_nested_dict(cfg), default=str)


def load_config(config_like: str | dict[str, Any]):
    if isinstance(config_like, str):
        config_dict = json.loads(config_like)
    elif isinstance(config_like, dict):
        config_dict = config_like
    else:
        raise TypeError(f"Unsupported config type: {type(config_like)}")
    return config_from_dict(config_dict)


def load_config_from_yaml(path: str | Path, config_key: str | None = None) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    candidates: dict[str, Any] = raw.get("experiments", raw)
    if config_key:
        config_dict = candidates.get(config_key)
        if config_dict is None:
            raise KeyError(f"Config '{config_key}' not found in {path}")
    else:
        if isinstance(raw.get("experiments"), dict):
            raise ValueError("config_key must be provided when the YAML file defines multiple experiments")
        config_dict = candidates
    if not isinstance(config_dict, dict):
        raise TypeError(f"Invalid config structure in {path}, expected a mapping")
    base_cfg = _load_base_cfg(config_dict.pop("base", raw.get("base", None)))
    return config_from_dict(config_dict, base_cfg=base_cfg)


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
