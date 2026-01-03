# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Convert sample_efficient_gpt imu_1 checkpoints into HF/vLLM format with
safe serialization and tokenizer export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, Imu1ForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_ROOT = Path("/home/george/sample_efficient_gpt")


CHAT_TEMPLATE = """
{{- '<|bos|>' -}}
{%- set ns = namespace(first_system='', injected=false) -%}
{%- if messages and messages[0].role == 'system' -%}
    {%- set ns.first_system = messages[0].content -%}
{%- endif -%}

{%- for message in messages -%}
    {%- if message.content is string -%}
        {%- set content = message.content -%}
    {%- else -%}
        {%- set content = '' -%}
    {%- endif -%}

    {%- if message.role == "user" -%}
        {%- if ns.first_system and not ns.injected -%}
            {{- '<|user_start|>' + ns.first_system + '\n\n' + content + '<|user_end|>' -}}
            {%- set ns.injected = true -%}
        {%- else -%}
            {{- '<|user_start|>' + content + '<|user_end|>' -}}
        {%- endif -%}

    {%- elif message.role == "assistant" -%}
        {{- '<|assistant_start|>' -}}
        {% generation -%}
        {{- content + '<|assistant_end|>' -}}
        {%- endgeneration %}
    {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
    {{- '<|assistant_start|>' -}}
{%- else -%}
    {% generation -%}
    {{- '<|endoftext|>' -}}
    {%- endgeneration %}
{%- endif -%}
"""


def _save_tokenizer(tokenizer_dir: Path, save_dir: Path) -> None:
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(save_dir)


def _patch_cpu_qknorm() -> None:
    """Monkey patch the custom sdpa qk-norm kernel for CPU testing."""
    from sample_efficient_gpt.transformer.attention import SelfDotProductAttnQKNorm

    def _cpu_forward(self, q, k, v, is_causal: bool = True, q_start: int = 0):
        qf, kf = q.float(), k.float()
        alpha = torch.rsqrt(qf.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        beta = torch.rsqrt(kf.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        qn = qf * alpha
        kn = kf * beta

        score = torch.bmm(qn, kn.transpose(1, 2)) * self.gain.to(qf)
        if is_causal:
            seq_q, seq_k = score.shape[-2:]
            q_pos = torch.arange(seq_q, device=score.device) + q_start
            k_pos = torch.arange(seq_k, device=score.device)
            causal = q_pos[:, None] >= k_pos[None, :]
            score = score.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(score, dim=-1)
        out = torch.bmm(probs, v.float())
        return out.to(q.dtype)

    SelfDotProductAttnQKNorm.forward = _cpu_forward  # type: ignore[method-assign]


def _load_checkpoint(ckpt_path: Path) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt or "config" not in ckpt:
        raise ValueError("Checkpoint does not contain expected keys 'model'/'config'.")
    if isinstance(ckpt["config"], str):
        ckpt["config"] = json.loads(ckpt["config"])
    return ckpt


def _resolve_tokens(
    tokenizer_dir: Path | None,
) -> tuple[int | None, int | None, int | None]:
    """Resolve (bos_id, eos_id, pad_id), optionally from a tokenizer directory."""
    bos_id: int | None = None
    eos_id: int | None = 256
    pad_id: int | None = None
    if tokenizer_dir is None:
        return bos_id, eos_id, pad_id
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
        bos_id = tok.bos_token_id or tok.convert_tokens_to_ids("<|bos|>")
        eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("<|endoftext|>")
        pad_id = tok.pad_token_id
    except Exception as exc:  # pragma: no cover
        print(f"[imu1] Warning: failed to load tokenizer from {tokenizer_dir}: {exc}")
    return bos_id, eos_id, pad_id


def _build_config(
    cfg_dict: dict[str, Any],
    tokenizer_dir: Path | None,
    tie_word_embeddings: bool | None = None,
):
    # Use the Transformers-side config so the converted checkpoint loads directly with HF APIs.
    from transformers.models.imu_1 import Imu1Config

    model_cfg = cfg_dict["model"]
    data_cfg = cfg_dict.get("data", {})
    max_pos = max(8192, data_cfg.get("context_length", 0), model_cfg.get("max_seq_len", 0))

    bos_id, eos_id, pad_id = _resolve_tokens(tokenizer_dir)

    if tie_word_embeddings is None:
        tie_word_embeddings = model_cfg.get("weight_tying", False)

    config = Imu1Config(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["d_model"],
        intermediate_size=model_cfg["d_ff"],
        num_hidden_layers=model_cfg["n_layers"],
        num_attention_heads=model_cfg["n_heads"],
        num_key_value_heads=model_cfg.get("n_kv_heads", model_cfg["n_heads"]),
        hidden_act="silu",
        max_position_embeddings=max_pos,
        rms_norm_eps=1e-5,
        rope_theta=model_cfg.get("theta", 10000),
        attn_qknorm=model_cfg.get("attn_qknorm", True),
        attn_val_residual=model_cfg.get("attn_val_residual", True),
        attn_gating=model_cfg.get("attn_gating", False),
        layernorm_scaling=model_cfg.get("layernorm_scaling", False),
        tie_word_embeddings=tie_word_embeddings,
        pad_token_id=data_cfg.get("pad_token_id", pad_id),
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )
    config.architectures = ["Imu1ForCausalLM"]
    return config


def _rename_key(name: str, layer_idx: int | None) -> str:
    prefix = f"model.layers.{layer_idx}" if layer_idx is not None else "model"
    if name == "embedding.weight":
        return "model.embed_tokens.weight"
    if name == "final_norm.gain":
        return "model.norm.weight"
    if name == "lm_head.linear.weight" or name == "lm_head.weight":
        return "lm_head.weight"
    if name == "lm_head.weight":
        return "lm_head.weight"

    if name.startswith("ln1.gain"):
        return f"{prefix}.input_layernorm.weight"
    if name.startswith("ln2.gain"):
        return f"{prefix}.post_attention_layernorm.weight"
    if name.startswith("attn.qkv.linear.weight"):
        return f"{prefix}.self_attn.qkv_proj.weight"
    if name.startswith("attn.out.linear.weight"):
        return f"{prefix}.self_attn.o_proj.weight"
    if name.startswith("attn.attn_gate.linear.weight"):
        return f"{prefix}.self_attn.attn_gate.weight"
    if name.startswith("attn.sdpa_qknorm.gain"):
        return f"{prefix}.self_attn.qk_norm.gain"
    if name.startswith("attn.alpha1"):
        return f"{prefix}.self_attn.alpha1"
    if name.startswith("attn.alpha2"):
        return f"{prefix}.self_attn.alpha2"
    if name.startswith("attn.scale"):
        return f"{prefix}.self_attn.value_scale"
    if name.startswith("ffn.up.linear.weight"):
        return f"{prefix}.mlp.up_proj.weight"
    if name.startswith("ffn.down.linear.weight"):
        return f"{prefix}.mlp.down_proj.weight"
    raise KeyError(f"Unhandled parameter name: {name}")


def _convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_state_dict: dict[str, torch.Tensor] = {}
    has_linear_head = "lm_head.linear.weight" in state_dict
    for name, tensor in state_dict.items():
        if name == "lm_head.weight" and has_linear_head:
            continue
        if name.startswith("blocks."):
            parts = name.split(".")
            layer_idx = int(parts[1])
            suffix = ".".join(parts[2:])
            new_key = _rename_key(suffix, layer_idx)
        else:
            new_key = _rename_key(name, None)
        new_state_dict[new_key] = tensor
    return new_state_dict


def _reverse_convert_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    reverse_sd: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name.startswith("model.layers."):
            _, _, layer_str, *rest = name.split(".")
            suffix = ".".join(rest)
            prefix = f"blocks.{layer_str}"
            if suffix.startswith("input_layernorm.rms_norm.weight") or suffix.startswith("input_layernorm.weight"):
                reverse_sd[f"{prefix}.ln1.gain"] = tensor
            elif suffix.startswith("post_attention_layernorm.rms_norm.weight") or suffix.startswith(
                "post_attention_layernorm.weight"
            ):
                reverse_sd[f"{prefix}.ln2.gain"] = tensor
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
            elif suffix.startswith("mlp.up_proj.weight"):
                reverse_sd[f"{prefix}.ffn.up.linear.weight"] = tensor
            elif suffix.startswith("mlp.down_proj.weight"):
                reverse_sd[f"{prefix}.ffn.down.linear.weight"] = tensor
            else:
                raise KeyError(f"Unhandled converted key: {name}")
        elif name == "model.embed_tokens.weight":
            reverse_sd["embedding.weight"] = tensor
        elif name == "model.norm.weight":
            reverse_sd["final_norm.gain"] = tensor
        elif name == "lm_head.weight":
            reverse_sd["lm_head.linear.weight"] = tensor
            reverse_sd["lm_head.weight"] = tensor
        else:
            raise KeyError(f"Unhandled converted key: {name}")
    if "embedding.weight" in reverse_sd and "lm_head.weight" in reverse_sd:
        reverse_sd["lm_head.weight"] = reverse_sd["embedding.weight"]
    return reverse_sd


def _run_ref_logits(cfg_dict: dict[str, Any], state_dict: dict[str, torch.Tensor], tokens: torch.Tensor):
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.cuda.set_device = lambda *_, **__: None  # type: ignore[attr-defined]
    torch.cuda.is_available = lambda: False  # type: ignore[attr-defined]
    _patch_cpu_qknorm()
    from sample_efficient_gpt.transformer.transformer import Transformer

    m_cfg = cfg_dict["model"]
    model = Transformer(
        m_cfg["n_layers"],
        m_cfg["vocab_size"],
        m_cfg["d_model"],
        m_cfg["n_heads"],
        m_cfg["d_ff"],
        attn_qknorm=m_cfg.get("attn_qknorm", False),
        attn_val_residual=m_cfg.get("attn_val_residual", False),
        attn_gating=m_cfg.get("attn_gating", False),
        layernorm_scaling=m_cfg.get("layernorm_scaling", False),
        theta=m_cfg.get("theta", 10000),
        n_kv_heads=m_cfg.get("n_kv_heads", m_cfg["n_heads"]),
        device="cpu",
        dtype=torch.float32,
        weight_tying=m_cfg.get("weight_tying", False),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    with torch.no_grad():
        logits, _ = model(tokens)
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/harddrive/checkpoints/1117_main_run_mix_decay/1660000.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("imu1_converted"))
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Optional tokenizer directory to pick BOS/EOS/PAD ids from (e.g., mix_bpe_hf2).",
    )
    parser.add_argument("--test", action="store_true", help="Run a small equivalence test on CPU.")
    parser.add_argument("--test-seq-len", type=int, default=8)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    sys.path.insert(0, str(REPO_ROOT))
    if args.sample_root.exists():
        sys.path.insert(0, str(args.sample_root))

    ckpt = _load_checkpoint(args.checkpoint)
    cfg_dict = ckpt["config"]
    state_dict = ckpt["model"]

    converted_sd = _convert_state_dict(state_dict)

    tie_override: bool | None = None
    if cfg_dict.get("model", {}).get("weight_tying", False):
        emb = state_dict.get("embedding.weight")
        head = state_dict.get("lm_head.linear.weight")
        if emb is not None and head is not None and not torch.equal(emb, head):
            tie_override = False
            print("[imu1] Warning: weight_tying=true but embeddings differ; disabling HF tie_word_embeddings.")

    config = _build_config(cfg_dict, args.tokenizer_dir, tie_override)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = Imu1ForCausalLM(config)
    model.load_state_dict(converted_sd, strict=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    if args.tokenizer_dir:
        _save_tokenizer(args.tokenizer_dir, args.output_dir)

    if args.test:
        torch.manual_seed(0)
        vocab = cfg_dict["model"]["vocab_size"]
        seq_len = min(args.test_seq_len, cfg_dict["data"].get("context_length", args.test_seq_len))
        tokens = torch.randint(0, vocab, (1, seq_len))

        logits_ref = _run_ref_logits(cfg_dict, state_dict, tokens)

        roundtrip_sd = _reverse_convert_state_dict(converted_sd)
        logits_roundtrip = _run_ref_logits(cfg_dict, roundtrip_sd, tokens)

        max_diff = (logits_ref - logits_roundtrip).abs().max().item()
        print(f"[imu1] Test max logits diff: {max_diff:.6f}")
        if max_diff > 1e-5:
            raise RuntimeError("Round-trip logits mismatch exceeds tolerance.")

    print(f"Saved converted weights and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
