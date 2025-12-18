# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Convert HF/vLLM Imu1 checkpoints back into sample_efficient_gpt imu_1
checkpoint format:

  {
      "model": <state_dict in blocks.* layout>,
      "config": {
          "model": {...},
          "data": {...},  # optional
          ...
      }
  }

Usage example:

  python convert_hf_to_imu.py \
      --hf-dir imu1_converted \
      --output /mnt/harddrive/checkpoints/from_hf/imu1_from_hf.pt

Optionally, you can provide a template training config (e.g. the original
JSON config used for training) and this script will just overwrite the
`config["model"]` section from the HF config:

  python convert_hf_to_sample_imu1.py \
      --hf-dir imu1_converted \
      --output imu1_from_hf.pt \
      --template-config /path/to/original_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import Imu1ForCausalLM, Imu1Config


def _reverse_convert_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Convert HF Imu1ForCausalLM state_dict back into sample_efficient_gpt layout.

    This is the inverse of _convert_state_dict in your forward converter.
    It is slightly more permissive: unknown HF keys are skipped instead of
    raising, so buffers like rotary_emb.inv_freq won't break conversion.
    """
    reverse_sd: Dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        # Per-layer weights
        if name.startswith("model.layers."):
            parts = name.split(".")
            # Expected: model.layers.{idx}.<suffix...>
            #           0    1      2     3..N
            if len(parts) < 4:
                continue  # unexpected, but ignore gracefully

            _, _, layer_str, *rest = parts
            suffix = ".".join(rest)
            prefix = f"blocks.{layer_str}"

            # LayerNorm 1
            if (
                suffix.startswith("input_layernorm.rms_norm.weight")
                or suffix.startswith("input_layernorm.weight")
            ):
                reverse_sd[f"{prefix}.ln1.gain"] = tensor

            # LayerNorm 2
            elif (
                suffix.startswith("post_attention_layernorm.rms_norm.weight")
                or suffix.startswith("post_attention_layernorm.weight")
            ):
                reverse_sd[f"{prefix}.ln2.gain"] = tensor

            # Attention projections
            elif suffix.startswith("self_attn.qkv_proj.weight"):
                reverse_sd[f"{prefix}.attn.qkv.linear.weight"] = tensor
            elif suffix.startswith("self_attn.o_proj.weight"):
                reverse_sd[f"{prefix}.attn.out.linear.weight"] = tensor
            elif suffix.startswith("self_attn.attn_gate.weight"):
                reverse_sd[f"{prefix}.attn.attn_gate.linear.weight"] = tensor
            elif suffix.startswith("self_attn.qk_norm.gain"):
                reverse_sd[f"{prefix}.attn.sdpa_qknorm.gain"] = tensor
            elif suffix.startswith("self_attn.alpha1"):
                reverse_sd[f"{prefix}.attn.alpha1"] = tensor
            elif suffix.startswith("self_attn.alpha2"):
                reverse_sd[f"{prefix}.attn.alpha2"] = tensor
            elif suffix.startswith("self_attn.value_scale"):
                reverse_sd[f"{prefix}.attn.scale"] = tensor

            # MLP projections
            elif suffix.startswith("mlp.up_proj.weight"):
                reverse_sd[f"{prefix}.ffn.up.linear.weight"] = tensor
            elif suffix.startswith("mlp.down_proj.weight"):
                reverse_sd[f"{prefix}.ffn.down.linear.weight"] = tensor

            # Unknown per-layer keys (e.g. buffers) are ignored
            else:
                continue

        # Embeddings / final norm / lm_head
        elif name == "model.embed_tokens.weight":
            reverse_sd["embedding.weight"] = tensor
        elif name == "model.norm.weight":
            reverse_sd["final_norm.gain"] = tensor
        elif name == "lm_head.weight":
            reverse_sd["lm_head.linear.weight"] = tensor

        # Everything else (e.g. rotary_emb.inv_freq buffers) is ignored
        else:
            continue

    return reverse_sd


def _build_sample_model_cfg(hf_cfg: Imu1Config) -> Dict[str, Any]:
    """
    Build the `config["model"]` section used by sample_efficient_gpt
    from an HF Imu1Config. This mirrors your _build_config mapping.
    """
    model_cfg: Dict[str, Any] = {
        "vocab_size": hf_cfg.vocab_size,
        "d_model": hf_cfg.hidden_size,
        "d_ff": hf_cfg.intermediate_size,
        "n_layers": hf_cfg.num_hidden_layers,
        "n_heads": hf_cfg.num_attention_heads,
        "n_kv_heads": getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads),
        "theta": getattr(hf_cfg, "rope_theta", 10000),
        "attn_qknorm": getattr(hf_cfg, "attn_qknorm", True),
        "attn_val_residual": getattr(hf_cfg, "attn_val_residual", True),
        "attn_gating": getattr(hf_cfg, "attn_gating", False),
        "layernorm_scaling": getattr(hf_cfg, "layernorm_scaling", False),
        "weight_tying": getattr(hf_cfg, "tie_word_embeddings", False),
        "max_seq_len": getattr(hf_cfg, "max_position_embeddings", 8192),
    }
    return model_cfg


def _load_template_config(path: Path) -> Dict[str, Any]:
    """
    Load a template config that you used for training (optional).

    - If `path.suffix == ".pt"`, it will load a torch checkpoint and use its
      "config" field.
    - Otherwise, it will treat `path` as a JSON file.
    """
    if path.suffix == ".pt":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config")
        if cfg is None:
            raise ValueError("Template checkpoint does not contain 'config' field.")
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        return cfg

    # Assume JSON
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_full_config(
    hf_cfg: Imu1Config,
    template_cfg_path: Path | None,
) -> Dict[str, Any]:
    """
    Build the full config dict to store in the sample_efficient_gpt checkpoint.

    If template_cfg_path is provided, we:
      - load the template config
      - overwrite template["model"] with values inferred from HF config

    Otherwise, we build a minimal config:
      {
          "model": { ...mapped from HF... },
          "data": {}
      }
    """
    model_cfg = _build_sample_model_cfg(hf_cfg)

    if template_cfg_path is not None:
        base_cfg = _load_template_config(template_cfg_path)
        # Ensure "model" section exists and overwrite it with HF-derived values
        base_cfg.setdefault("model", {})
        base_cfg["model"].update(model_cfg)
        # Optionally, ensure there is at least a "data" section
        base_cfg.setdefault("data", {})
        return base_cfg

    # Minimal config if you don't care about optimizer / training params
    return {
        "model": model_cfg,
        "data": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-dir",
        type=Path,
        required=True,
        help="Path to HF Imu1 checkpoint directory (containing config.json + safetensors).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output .pt checkpoint in sample_efficient_gpt format.",
    )
    parser.add_argument(
        "--template-config",
        type=Path,
        default=None,
        help=(
            "Optional path to a template config (JSON or .pt checkpoint) whose "
            "'config[\"model\"]' section will be replaced with values inferred "
            "from the HF config."
        ),
    )
    parser.add_argument(
        "--no-strict-dtype",
        action="store_true",
        help=(
            "If set, the HF model is loaded with default dtypes (whatever is stored "
            "in safetensors). By default we force torch.float32 on CPU."
        ),
    )
    args = parser.parse_args()

    # 1. Load HF config and model
    hf_cfg = Imu1Config.from_pretrained(args.hf_dir)

    if args.no_strict_dtype:
        model = Imu1ForCausalLM.from_pretrained(args.hf_dir)
    else:
        # Force CPU + float32 to be fully portable as a .pt checkpoint.
        model = Imu1ForCausalLM.from_pretrained(
            args.hf_dir,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )

    model.eval()
    hf_state = model.state_dict()

    # 2. Convert HF state dict back into sample_efficient_gpt layout
    sample_state = _reverse_convert_state_dict(hf_state)

    # 3. Build config dict in your original format
    cfg_dict = _build_full_config(hf_cfg, args.template_config)

    # 4. Save checkpoint
    ckpt = {
        "model": sample_state,
        "config": cfg_dict,
        "iteration": 0
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)

    print(f"[imu1] Saved sample_efficient_gpt checkpoint to: {args.output}")


if __name__ == "__main__":
    main()