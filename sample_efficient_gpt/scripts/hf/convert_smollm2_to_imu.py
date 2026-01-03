"""
Convert an HF SmolLM2 (LLaMA) checkpoint into the sample_efficient_gpt "imu_1"
checkpoint format.

This script is intended for *shape-changing* conversions (e.g. widening), while
keeping layer count intact by default.

It:
  - loads the HF model (LlamaForCausalLM / AutoModelForCausalLM)
  - builds a sample_efficient_gpt Transformer from a template YAML/JSON/.pt config
    (or from the HF config if no template is provided)
  - maps weights:
      - q_proj/k_proj/v_proj -> blocks.*.attn.qkv.linear.weight (native GQA: q has n_heads, k/v have n_kv_heads)
      - gate_proj + up_proj  -> blocks.*.ffn.up.linear.weight (SwiGLU packed)
      - o_proj               -> blocks.*.attn.out.linear.weight
      - down_proj            -> blocks.*.ffn.down.linear.weight
      - norms/embeddings/head
  - when target tensors are larger than source tensors, it preserves the model's
    muP init but scales it by `--init-multiplier` before copying the source slice.

Example:
  source /home/george/axolotl/.venv/bin/activate
  python convert_smollm2_to_imu.py \\
    --hf-dir HuggingFaceTB/SmolLM2-135M \\
    --template-config ../../configs/main_runs.yaml --config-key main_run_mix \\
    --output /mnt/harddrive/checkpoints/from_hf/smollm2_to_imu.pt \\
    --n-layers from-hf --vocab-size from-hf --width 576 --n-heads 9 --d-ff 1536
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoConfig, AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_template_config(path: Path, config_key: str | None) -> dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        from sample_efficient_gpt.utils.config_tools import dataclass_to_nested_dict, load_config_from_yaml

        cfg = load_config_from_yaml(path, config_key)
        return dataclass_to_nested_dict(cfg)

    if path.suffix == ".pt":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config")
        if cfg is None:
            raise ValueError("Template checkpoint does not contain a 'config' field.")
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        if not isinstance(cfg, dict):
            raise TypeError(f"Unsupported template config type inside .pt: {type(cfg)}")
        return cfg

    if path.suffix == ".json":
        return json.loads(path.read_text())

    raise ValueError(f"Unsupported template config format: {path}")


def _parse_from_hf_or_int(value: str, hf_value: int) -> int:
    if value == "from-hf":
        return int(hf_value)
    raise ValueError(f"Expected 'from-hf' or an int, got: {value}")


def _parse_from_hf_or_float(value: str, hf_value: float) -> float:
    if value == "from-hf":
        return float(hf_value)
    raise ValueError(f"Expected 'from-hf' or a float, got: {value}")


def _parse_from_sources_or_int(value: str, *, hf_value: int, template_value: int | None) -> int:
    if value == "from-hf":
        return int(hf_value)
    if value == "from-template":
        return int(template_value if template_value is not None else hf_value)
    return int(value)


def _parse_from_sources_or_float(value: str, *, hf_value: float, template_value: float | None) -> float:
    if value == "from-hf":
        return float(hf_value)
    if value == "from-template":
        return float(template_value if template_value is not None else hf_value)
    return float(value)


def _scale_and_copy_like(dst: torch.Tensor, src: torch.Tensor, init_multiplier: float) -> torch.Tensor:
    """
    Returns a new tensor shaped like dst:
      - starts from dst (muP init)
      - scales by init_multiplier (so newly-initialized regions become smaller)
      - copies the overlapping slice from src
    """
    out = dst.detach().clone()
    if init_multiplier != 1.0:
        out.mul_(float(init_multiplier))

    # Copy overlap (supports 1D and 2D tensors in this repo's state_dict).
    if out.ndim == 1 and src.ndim == 1:
        n = min(out.shape[0], src.shape[0])
        out[:n] = src[:n].to(out.dtype)
        return out

    if out.ndim == 2 and src.ndim == 2:
        r = min(out.shape[0], src.shape[0])
        c = min(out.shape[1], src.shape[1])
        out[:r, :c] = src[:r, :c].to(out.dtype)
        return out

    raise ValueError(f"Unsupported resize: dst{tuple(out.shape)} src{tuple(src.shape)}")


WideningMode = Literal["noise", "preserve", "preserve-norm"]


def _init_tensor_like(dst: torch.Tensor, *, init_multiplier: float, mode: WideningMode) -> torch.Tensor:
    if mode == "preserve":
        return torch.zeros_like(dst)
    out = dst.detach().clone()
    if init_multiplier != 1.0:
        out.mul_(float(init_multiplier))
    return out


def _copy_overlap_1d(out: torch.Tensor, src: torch.Tensor) -> None:
    n = min(out.shape[0], src.shape[0])
    out[:n] = src[:n].to(out.dtype)


def _copy_overlap_2d(out: torch.Tensor, src: torch.Tensor) -> None:
    r = min(out.shape[0], src.shape[0])
    c = min(out.shape[1], src.shape[1])
    out[:r, :c] = src[:r, :c].to(out.dtype)


def _expand_gqa_kv_to_mha(kv_weight: torch.Tensor, *, n_kv_heads: int, n_heads: int, head_dim: int) -> torch.Tensor:
    """
    Legacy helper for expanding GQA KV weights into an MHA layout by repeating KV heads.
    Kept for reference; native GQA is now supported in sample_efficient_gpt, so the main
    conversion path does not rely on this.
    """
    if n_kv_heads == n_heads:
        return kv_weight
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"Cannot expand GQA KV: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}")

    out_dim, in_dim = kv_weight.shape
    expected_out = n_kv_heads * head_dim
    if out_dim != expected_out:
        raise ValueError(f"KV weight out_dim mismatch: got {out_dim}, expected {expected_out}")

    kv = kv_weight.view(n_kv_heads, head_dim, in_dim)
    repeat = n_heads // n_kv_heads
    kv = kv.repeat_interleave(repeat, dim=0)  # (n_heads, head_dim, in_dim)
    return kv.reshape(n_heads * head_dim, in_dim)


def _build_qkv_weight(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    src_hidden: int,
    dst_hidden: int,
    src_kv_out: int,
    dst_kv_out: int,
    init_qkv: torch.Tensor,
    init_multiplier: float,
    mode: WideningMode,
) -> torch.Tensor:
    """
    Pack (q,k,v) into one qkv weight, and resize into the destination qkv shape.
    """
    out = _init_tensor_like(init_qkv, init_multiplier=init_multiplier, mode=mode)

    col_overlap = min(src_hidden, dst_hidden)

    q_rows = min(src_hidden, dst_hidden)
    kv_rows = min(src_kv_out, dst_kv_out)

    # Q block: (dst_hidden, dst_hidden) top-left
    out[0:q_rows, 0:col_overlap] = q[:q_rows, :col_overlap].to(out.dtype)
    # K block: after Q
    k_row0 = dst_hidden
    out[k_row0 : k_row0 + kv_rows, 0:col_overlap] = k[:kv_rows, :col_overlap].to(out.dtype)
    # V block: after K
    v_row0 = dst_hidden + dst_kv_out
    out[v_row0 : v_row0 + kv_rows, 0:col_overlap] = v[:kv_rows, :col_overlap].to(out.dtype)

    return out


def _build_swiglu_up_weight(
    *,
    gate: torch.Tensor,
    up: torch.Tensor,
    src_hidden: int,
    dst_hidden: int,
    src_dff: int,
    dst_dff: int,
    init_up: torch.Tensor,
    init_multiplier: float,
    mode: WideningMode,
) -> torch.Tensor:
    """
    sample_efficient_gpt SwiGLU expects a single Linear(d_model -> 2*d_ff) where
    the first half is the gate projection and the second half is the up projection.

    HF LLaMA provides separate (gate_proj, up_proj), both shaped (src_dff, src_hidden).
    This function places them into the correct halves even when dst_dff != src_dff.
    """
    out = _init_tensor_like(init_up, init_multiplier=init_multiplier, mode=mode)

    r_hidden = min(src_hidden, dst_hidden)
    r_dff = min(src_dff, dst_dff)

    # gate half
    out[0:r_dff, 0:r_hidden] = gate[:r_dff, :r_hidden].to(out.dtype)
    # up half
    out[dst_dff : dst_dff + r_dff, 0:r_hidden] = up[:r_dff, :r_hidden].to(out.dtype)

    return out


def _make_repeat_map(src_units: int, dst_units: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      old_idx: (dst_units,) long tensor mapping each new unit -> an old unit
      counts:  (src_units,) long tensor with how many times each old unit is repeated
    """
    if src_units <= 0 or dst_units <= 0:
        raise ValueError(f"Invalid units: src_units={src_units}, dst_units={dst_units}")
    base = dst_units // src_units
    rem = dst_units % src_units
    counts = torch.full((src_units,), base, dtype=torch.long)
    if rem:
        counts[:rem] += 1
    old_idx = torch.arange(src_units, dtype=torch.long).repeat_interleave(counts)
    if old_idx.numel() != dst_units:
        raise RuntimeError("Repeat map construction failed.")
    return old_idx, counts


def _build_swiglu_net2wider(
    *,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    src_hidden: int,
    dst_hidden: int,
    src_dff: int,
    dst_dff: int,
    init_up: torch.Tensor,
    init_down: torch.Tensor,
    init_multiplier: float,
    mode: WideningMode,
    down_noise_std: float,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Net2Wider-style FFN widening on the intermediate dimension.
    """
    if dst_dff < src_dff:
        raise ValueError("net2wider FFN expects dst_dff >= src_dff")

    out_up = _init_tensor_like(init_up, init_multiplier=init_multiplier, mode=mode)
    out_down = _init_tensor_like(init_down, init_multiplier=init_multiplier, mode=mode)

    r_in = min(src_hidden, dst_hidden)
    r_out = min(src_hidden, dst_hidden)

    old_idx, counts = _make_repeat_map(src_dff, dst_dff)
    denom = counts[old_idx].to(dtype=out_down.dtype)

    out_up[:dst_dff, :r_in] = gate[old_idx, :r_in].to(out_up.dtype)
    out_up[dst_dff : 2 * dst_dff, :r_in] = up[old_idx, :r_in].to(out_up.dtype)

    down_src = down[:r_out, :src_dff].to(out_down.dtype)
    down_new = down_src[:, old_idx] / denom.unsqueeze(0)

    if down_noise_std > 0.0:
        sigma = down_src.std(dim=0, unbiased=False)
        pos = 0
        for j in range(src_dff):
            cnt = int(counts[j].item())
            if cnt <= 1:
                pos += cnt
                continue
            n = torch.randn((r_out, cnt), generator=rng, dtype=down_new.dtype)
            n.mul_(float(down_noise_std) * sigma[j].to(down_new.dtype) / float(cnt))
            n.sub_(n.mean(dim=1, keepdim=True))
            down_new[:, pos : pos + cnt] += n
            pos += cnt

    out_down[:r_out, :dst_dff] = down_new
    return out_up, out_down


def _build_qkv_weight_net2wider_blockrepeat(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    src_hidden: int,
    dst_hidden: int,
    src_heads: int,
    dst_heads: int,
    src_kv_heads: int,
    dst_kv_heads: int,
    head_dim: int,
    init_qkv: torch.Tensor,
    init_multiplier: float,
    mode: WideningMode,
) -> torch.Tensor:
    """
    Net2Wider attention widening for packed QKV via block repetition.
    """
    out = _init_tensor_like(init_qkv, init_multiplier=init_multiplier, mode=mode)
    col_overlap = min(src_hidden, dst_hidden)

    if dst_heads % src_heads != 0 or dst_kv_heads % src_kv_heads != 0:
        raise ValueError("net2wider_blockrepeat requires integer head scaling for both Q and KV heads")
    f_q = dst_heads // src_heads
    f_kv = dst_kv_heads // src_kv_heads
    if f_q != f_kv:
        raise ValueError("net2wider_blockrepeat requires the same scale factor for heads and kv_heads")

    for nh in range(dst_heads):
        oh = nh % src_heads
        dst_r = slice(nh * head_dim, (nh + 1) * head_dim)
        src_r = slice(oh * head_dim, (oh + 1) * head_dim)
        out[dst_r, :col_overlap] = q[src_r, :col_overlap].to(out.dtype)

    dst_kv_out = dst_kv_heads * head_dim
    src_kv_out = src_kv_heads * head_dim

    k_row0 = dst_hidden
    for nk in range(dst_kv_heads):
        ok = nk % src_kv_heads
        dst_r = slice(k_row0 + nk * head_dim, k_row0 + (nk + 1) * head_dim)
        src_r = slice(ok * head_dim, (ok + 1) * head_dim)
        out[dst_r, :col_overlap] = k[src_r, :col_overlap].to(out.dtype)

    v_row0 = dst_hidden + dst_kv_out
    for nk in range(dst_kv_heads):
        ok = nk % src_kv_heads
        dst_r = slice(v_row0 + nk * head_dim, v_row0 + (nk + 1) * head_dim)
        src_r = slice(ok * head_dim, (ok + 1) * head_dim)
        out[dst_r, :col_overlap] = v[src_r, :col_overlap].to(out.dtype)

    return out


def _build_attn_out_weight_net2wider_blockrepeat(
    *,
    o: torch.Tensor,
    src_hidden: int,
    dst_hidden: int,
    src_heads: int,
    dst_heads: int,
    head_dim: int,
    init_out: torch.Tensor,
    init_multiplier: float,
    mode: WideningMode,
    out_noise_std: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """
    Net2Wider for attention output projection via head block repetition.
    """
    out = _init_tensor_like(init_out, init_multiplier=init_multiplier, mode=mode)
    row_overlap = min(src_hidden, dst_hidden)

    if dst_heads % src_heads != 0:
        raise ValueError("net2wider_blockrepeat requires integer head scaling")
    f = dst_heads // src_heads

    if f == 1:
        r = min(src_hidden, dst_hidden)
        c = min(src_hidden, dst_hidden)
        out[:r, :c] = o[:r, :c].to(out.dtype)
        return out

    o_src = o[:row_overlap, :src_hidden].to(out.dtype)
    sigma = o_src.std(dim=0, unbiased=False)

    for oh in range(src_heads):
        for j in range(head_dim):
            old_col = oh * head_dim + j
            base = o_src[:, old_col] / float(f)

            for b in range(f):
                nh = b * src_heads + oh
                new_col = nh * head_dim + j
                out[:row_overlap, new_col] = base

            if out_noise_std > 0.0:
                n = torch.randn((row_overlap, f), generator=rng, dtype=out.dtype)
                n.mul_(float(out_noise_std) * sigma[old_col].to(out.dtype) / float(f))
                n.sub_(n.mean(dim=1, keepdim=True))
                for b in range(f):
                    nh = b * src_heads + oh
                    new_col = nh * head_dim + j
                    out[:row_overlap, new_col] += n[:, b]

    return out


def _patch_cpu_safety() -> None:
    """
    Avoid unconditional CUDA side-effects at import time from some Triton modules.
    This mirrors the approach used in scripts/hf/convert_imu1_checkpoint.py.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.cuda.set_device = lambda *_, **__: None  # type: ignore[attr-defined]
    torch.cuda.is_available = lambda: False  # type: ignore[attr-defined]


def _convert(
    *,
    hf_dir: str,
    template_cfg: dict[str, Any] | None,
    output: Path,
    init_multiplier: float,
    widening_mode: WideningMode,
    width: int,
    n_heads: int,
    n_kv_heads: int,
    d_ff: int,
    n_layers: int,
    vocab_size: int,
    theta: float,
    attn_qknorm: bool,
    attn_val_residual: bool,
    attn_gating: bool | str,
    layernorm_scaling: bool,
    weight_tying: bool,
    qknorm_gain: float | None,
    ffn_expand: str,
    ffn_down_noise_std: float,
    attn_expand: str,
    attn_out_noise_std: float,
    expand_seed: int,
) -> None:
    hf_cfg = AutoConfig.from_pretrained(hf_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32, device_map={"": "cpu"})
    hf_model.eval()
    hf_sd = hf_model.state_dict()

    if getattr(hf_cfg, "model_type", None) != "llama":
        raise ValueError(f"Expected a LLaMA-like model, got model_type={getattr(hf_cfg, 'model_type', None)}")

    hf_layers = int(getattr(hf_cfg, "num_hidden_layers"))
    if n_layers > hf_layers:
        raise ValueError(f"Target n_layers={n_layers} exceeds HF n_layers={hf_layers}")

    src_hidden = int(getattr(hf_cfg, "hidden_size"))
    src_heads = int(getattr(hf_cfg, "num_attention_heads"))
    src_kv_heads = int(getattr(hf_cfg, "num_key_value_heads", src_heads))
    src_head_dim = int(getattr(hf_cfg, "head_dim", src_hidden // src_heads))
    dst_head_dim = width // n_heads
    print(f"[hf] hidden={src_hidden} heads={src_heads} kv_heads={src_kv_heads} head_dim={src_head_dim}")

    if widening_mode == "preserve":
        if width < src_hidden:
            raise ValueError("widening_mode=preserve requires width >= HF hidden_size")
        if d_ff < int(getattr(hf_cfg, "intermediate_size")):
            raise ValueError("widening_mode=preserve requires d_ff >= HF intermediate_size")
        if n_heads < src_heads:
            raise ValueError("widening_mode=preserve requires n_heads >= HF num_attention_heads")
        if dst_head_dim != src_head_dim:
            raise ValueError(
                "widening_mode=preserve requires matching head_dim "
                f"(dst width/n_heads={dst_head_dim} vs HF head_dim={src_head_dim})"
            )

    # Build target model (for correct shapes + muP init).
    _patch_cpu_safety()
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from sample_efficient_gpt.transformer.transformer import Transformer

    target = Transformer(
        n_layers=n_layers,
        vocab_size=vocab_size,
        d_model=width,
        n_heads=n_heads,
        d_ff=d_ff,
        n_kv_heads=n_kv_heads,
        attn_qknorm=attn_qknorm,
        attn_val_residual=attn_val_residual,
        attn_gating=attn_gating,
        layernorm_scaling=layernorm_scaling,
        theta=theta,
        device="cpu",
        dtype=torch.float32,
        weight_tying=weight_tying,
        num_grad_checkpoint_layers=0,
    )
    target.eval()
    out_sd: dict[str, torch.Tensor] = target.state_dict()

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(expand_seed))

    rmsnorm_gain_scale = (
        (src_hidden / width) ** 0.5 if (widening_mode in {"preserve", "preserve-norm"} and width != src_hidden) else 1.0
    )

    # Embedding / head / final norm
    if widening_mode == "preserve":
        emb = _init_tensor_like(out_sd["embedding.weight"], init_multiplier=init_multiplier, mode=widening_mode)
        _copy_overlap_2d(emb, hf_sd["model.embed_tokens.weight"])
        out_sd["embedding.weight"] = emb
    else:
        out_sd["embedding.weight"] = _scale_and_copy_like(
            out_sd["embedding.weight"], hf_sd["model.embed_tokens.weight"], init_multiplier
        )

    if "lm_head.weight" in hf_sd:
        if widening_mode == "preserve":
            head = _init_tensor_like(
                out_sd["lm_head.linear.weight"], init_multiplier=init_multiplier, mode=widening_mode
            )
            _copy_overlap_2d(head, hf_sd["lm_head.weight"])
            out_sd["lm_head.linear.weight"] = head
        else:
            out_sd["lm_head.linear.weight"] = _scale_and_copy_like(
                out_sd["lm_head.linear.weight"], hf_sd["lm_head.weight"], init_multiplier
            )
    else:
        # tie_word_embeddings sometimes removes lm_head from state dict; fall back to embeddings.
        if widening_mode == "preserve":
            head = _init_tensor_like(
                out_sd["lm_head.linear.weight"], init_multiplier=init_multiplier, mode=widening_mode
            )
            _copy_overlap_2d(head, hf_sd["model.embed_tokens.weight"])
            out_sd["lm_head.linear.weight"] = head
        else:
            out_sd["lm_head.linear.weight"] = _scale_and_copy_like(
                out_sd["lm_head.linear.weight"], hf_sd["model.embed_tokens.weight"], init_multiplier
            )

    if widening_mode in {"preserve", "preserve-norm"}:
        gain = _init_tensor_like(out_sd["final_norm.gain"], init_multiplier=init_multiplier, mode=widening_mode)
        _copy_overlap_1d(gain, hf_sd["model.norm.weight"])
        gain.mul_(float(rmsnorm_gain_scale))
        out_sd["final_norm.gain"] = gain
    else:
        out_sd["final_norm.gain"] = _scale_and_copy_like(
            out_sd["final_norm.gain"], hf_sd["model.norm.weight"], init_multiplier
        )

    # Per-layer conversion.
    for layer_idx in range(n_layers):
        hf_prefix = f"model.layers.{layer_idx}"
        imu_prefix = f"blocks.{layer_idx}"

        if widening_mode in {"preserve", "preserve-norm"}:
            ln1 = _init_tensor_like(
                out_sd[f"{imu_prefix}.ln1.gain"], init_multiplier=init_multiplier, mode=widening_mode
            )
            _copy_overlap_1d(ln1, hf_sd[f"{hf_prefix}.input_layernorm.weight"])
            ln1.mul_(float(rmsnorm_gain_scale))
            out_sd[f"{imu_prefix}.ln1.gain"] = ln1

            ln2 = _init_tensor_like(
                out_sd[f"{imu_prefix}.ln2.gain"], init_multiplier=init_multiplier, mode=widening_mode
            )
            _copy_overlap_1d(ln2, hf_sd[f"{hf_prefix}.post_attention_layernorm.weight"])
            ln2.mul_(float(rmsnorm_gain_scale))
            out_sd[f"{imu_prefix}.ln2.gain"] = ln2
        else:
            out_sd[f"{imu_prefix}.ln1.gain"] = _scale_and_copy_like(
                out_sd[f"{imu_prefix}.ln1.gain"],
                hf_sd[f"{hf_prefix}.input_layernorm.weight"],
                init_multiplier,
            )
            out_sd[f"{imu_prefix}.ln2.gain"] = _scale_and_copy_like(
                out_sd[f"{imu_prefix}.ln2.gain"],
                hf_sd[f"{hf_prefix}.post_attention_layernorm.weight"],
                init_multiplier,
            )

        q_w = hf_sd[f"{hf_prefix}.self_attn.q_proj.weight"]
        k_w = hf_sd[f"{hf_prefix}.self_attn.k_proj.weight"]
        v_w = hf_sd[f"{hf_prefix}.self_attn.v_proj.weight"]

        # q_proj is always full hidden; k/v are GQA (kv_heads * head_dim).
        if q_w.shape != (src_hidden, src_hidden):
            raise ValueError(f"Unexpected q_proj shape {tuple(q_w.shape)} vs ({src_hidden},{src_hidden})")
        if k_w.shape[1] != src_hidden or v_w.shape[1] != src_hidden:
            raise ValueError("Unexpected k/v proj in_features (expected hidden_size)")
        src_kv_out = int(src_kv_heads * src_head_dim)
        if k_w.shape[0] != src_kv_out or v_w.shape[0] != src_kv_out:
            raise ValueError(f"Unexpected k/v out_features (expected {src_kv_out})")

        dst_head_dim = width // n_heads
        dst_kv_out = int(n_kv_heads * dst_head_dim)

        qkv_key = f"{imu_prefix}.attn.qkv.linear.weight"
        if attn_expand == "net2wider":
            if dst_head_dim != src_head_dim:
                raise ValueError("attn_expand=net2wider requires matching head_dim")
            if (n_heads % src_heads) != 0 or (n_kv_heads % src_kv_heads) != 0:
                raise ValueError("attn_expand=net2wider requires integer scaling of heads and kv_heads")
            if (n_heads // src_heads) != (n_kv_heads // src_kv_heads):
                raise ValueError("attn_expand=net2wider requires the same scale factor for heads and kv_heads")

            out_sd[qkv_key] = _build_qkv_weight_net2wider_blockrepeat(
                q=q_w,
                k=k_w,
                v=v_w,
                src_hidden=src_hidden,
                dst_hidden=width,
                src_heads=src_heads,
                dst_heads=n_heads,
                src_kv_heads=src_kv_heads,
                dst_kv_heads=n_kv_heads,
                head_dim=src_head_dim,
                init_qkv=out_sd[qkv_key],
                init_multiplier=init_multiplier,
                mode=widening_mode,
            )

            out_sd[f"{imu_prefix}.attn.out.linear.weight"] = _build_attn_out_weight_net2wider_blockrepeat(
                o=hf_sd[f"{hf_prefix}.self_attn.o_proj.weight"],
                src_hidden=src_hidden,
                dst_hidden=width,
                src_heads=src_heads,
                dst_heads=n_heads,
                head_dim=src_head_dim,
                init_out=out_sd[f"{imu_prefix}.attn.out.linear.weight"],
                init_multiplier=init_multiplier,
                mode=widening_mode,
                out_noise_std=float(attn_out_noise_std),
                rng=rng,
            )
        else:
            out_sd[qkv_key] = _build_qkv_weight(
                q=q_w,
                k=k_w,
                v=v_w,
                src_hidden=src_hidden,
                dst_hidden=width,
                src_kv_out=src_kv_out,
                dst_kv_out=dst_kv_out,
                init_qkv=out_sd[qkv_key],
                init_multiplier=init_multiplier,
                mode=widening_mode,
            )

            if widening_mode == "preserve":
                out_w = _init_tensor_like(
                    out_sd[f"{imu_prefix}.attn.out.linear.weight"], init_multiplier=init_multiplier, mode=widening_mode
                )
                _copy_overlap_2d(out_w, hf_sd[f"{hf_prefix}.self_attn.o_proj.weight"])
                out_sd[f"{imu_prefix}.attn.out.linear.weight"] = out_w
            else:
                out_sd[f"{imu_prefix}.attn.out.linear.weight"] = _scale_and_copy_like(
                    out_sd[f"{imu_prefix}.attn.out.linear.weight"],
                    hf_sd[f"{hf_prefix}.self_attn.o_proj.weight"],
                    init_multiplier,
                )

        gate_w = hf_sd[f"{hf_prefix}.mlp.gate_proj.weight"]
        up_w = hf_sd[f"{hf_prefix}.mlp.up_proj.weight"]
        down_src = hf_sd[f"{hf_prefix}.mlp.down_proj.weight"]

        src_dff = gate_w.shape[0]
        if ffn_expand == "net2wider" and d_ff >= src_dff:
            up_new, down_new = _build_swiglu_net2wider(
                gate=gate_w,
                up=up_w,
                down=down_src,
                src_hidden=src_hidden,
                dst_hidden=width,
                src_dff=src_dff,
                dst_dff=d_ff,
                init_up=out_sd[f"{imu_prefix}.ffn.up.linear.weight"],
                init_down=out_sd[f"{imu_prefix}.ffn.down.linear.weight"],
                init_multiplier=init_multiplier,
                mode=widening_mode,
                down_noise_std=float(ffn_down_noise_std),
                rng=rng,
            )
            out_sd[f"{imu_prefix}.ffn.up.linear.weight"] = up_new
            out_sd[f"{imu_prefix}.ffn.down.linear.weight"] = down_new
        else:
            out_sd[f"{imu_prefix}.ffn.up.linear.weight"] = _build_swiglu_up_weight(
                gate=gate_w,
                up=up_w,
                src_hidden=src_hidden,
                dst_hidden=width,
                src_dff=src_dff,
                dst_dff=d_ff,
                init_up=out_sd[f"{imu_prefix}.ffn.up.linear.weight"],
                init_multiplier=init_multiplier,
                mode=widening_mode,
            )

            if widening_mode == "preserve":
                down_w = _init_tensor_like(
                    out_sd[f"{imu_prefix}.ffn.down.linear.weight"], init_multiplier=init_multiplier, mode=widening_mode
                )
                _copy_overlap_2d(down_w, down_src)
                out_sd[f"{imu_prefix}.ffn.down.linear.weight"] = down_w
            else:
                out_sd[f"{imu_prefix}.ffn.down.linear.weight"] = _scale_and_copy_like(
                    out_sd[f"{imu_prefix}.ffn.down.linear.weight"],
                    down_src,
                    init_multiplier,
                )

        # If gating is enabled, ensure the initial behavior is a no-op.
        # In attention.py we use `2*sigmoid(attn_gate(x))`, so zero weights => gate==1.
        if attn_gating:
            gate_key = f"{imu_prefix}.attn.attn_gate.linear.weight"
            if gate_key in out_sd:
                out_sd[gate_key] = torch.zeros_like(out_sd[gate_key])

        if attn_qknorm and qknorm_gain is not None:
            gain_key = f"{imu_prefix}.attn.sdpa_qknorm.gain"
            if gain_key in out_sd:
                out_sd[gain_key] = torch.tensor([float(qknorm_gain)], dtype=out_sd[gain_key].dtype)

    # Config to save with the checkpoint.
    if template_cfg is not None:
        cfg_dict = template_cfg
    else:
        cfg_dict = {"model": {}, "data": {}, "trainer": {}, "optim": {}}
    cfg_dict.setdefault("model", {})
    template_model = (cfg_dict or {}).get("model", {})
    cfg_dict["model"].update(
        {
            "vocab_size": vocab_size,
            "d_model": width,
            "d_ff": d_ff,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "theta": theta,
            "weight_tying": weight_tying,
            "attn_qknorm": attn_qknorm,
            "attn_val_residual": attn_val_residual,
            "attn_gating": attn_gating,
            "layernorm_scaling": layernorm_scaling,
        }
    )

    ckpt = {
        "model": out_sd,
        "config": cfg_dict,
        "iteration": 0,
        # Keep HF-specific info out of config dataclasses to avoid breaking config parsing.
        "hf_metadata": {
            "num_attention_heads": src_heads,
            "num_key_value_heads": src_kv_heads,
            "head_dim": src_head_dim,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output)

    print(f"[imu1] Saved checkpoint: {output}")
    print(
        "[imu1] Notes: copied SmolLM2 weights into muP-init model; "
        f"newly-initialized regions scaled by init_multiplier={init_multiplier}; widening_mode={widening_mode}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hf-dir", type=str, required=True, help="HF repo id or local path (e.g. HuggingFaceTB/SmolLM2-135M)."
    )
    p.add_argument(
        "--output", type=Path, required=True, help="Output .pt checkpoint path (sample_efficient_gpt format)."
    )

    p.add_argument(
        "--template-config",
        type=Path,
        default=None,
        help="Optional template config (.yaml/.yml/.json or a .pt checkpoint with a 'config' field).",
    )
    p.add_argument(
        "--config-key",
        type=str,
        default=None,
        help="Config key for YAML files that define multiple experiments (e.g. main_run_mix).",
    )

    p.add_argument(
        "--init-multiplier",
        type=float,
        default=1.0,
        help="Scales muP-initialized regions that are *not* covered by the source weights (e.g. when widening).",
    )
    p.add_argument(
        "--widening-mode",
        type=str,
        default="noise",
        choices=["noise", "preserve", "preserve-norm"],
        help=(
            "How to initialize expanded tensors when widening: "
            "'noise' keeps the model's muP init (scaled by --init-multiplier); "
            "'preserve' zero-fills expanded regions and rescales RMSNorm gains so the widened model is function-preserving; "
            "'preserve-norm' keeps muP init but rescales RMSNorm gains (often restores eval parity when widening)."
        ),
    )
    p.add_argument(
        "--ffn-expand",
        type=str,
        default="slice",
        choices=["slice", "net2wider"],
        help=(
            "FFN widening behavior when dst d_ff > src d_ff. "
            "'slice' keeps current behavior (copy overlap + init rest). "
            "'net2wider' duplicates FFN neurons and splits Wdown to preserve function."
        ),
    )
    p.add_argument(
        "--ffn-down-noise-std",
        type=float,
        default=0.0,
        help=(
            "If --ffn-expand=net2wider, add zero-sum Gaussian noise to duplicated Wdown columns "
            "(per original neuron group). This preserves the forward pass but breaks symmetry."
        ),
    )
    p.add_argument(
        "--attn-expand",
        type=str,
        default="slice",
        choices=["slice", "net2wider"],
        help=(
            "Attention widening when dst heads > src heads (and head_dim matches). "
            "'slice' keeps current behavior. "
            "'net2wider' duplicates heads and splits Wo columns to preserve function."
        ),
    )
    p.add_argument(
        "--attn-out-noise-std",
        type=float,
        default=0.0,
        help=(
            "If --attn-expand=net2wider, add zero-sum Gaussian noise to duplicated Wo columns "
            "(per original head-feature). Preserves forward pass but breaks symmetry."
        ),
    )
    p.add_argument(
        "--expand-seed",
        type=int,
        default=0,
        help="RNG seed used for net2wider symmetry-breaking noise (converter-time only).",
    )

    # Target architecture controls (defaults: template values if provided, else from HF).
    p.add_argument("--width", type=str, default="from-template", help="Target d_model, or 'from-template'/'from-hf'.")
    p.add_argument("--n-heads", type=str, default="from-template", help="Target n_heads, or 'from-template'/'from-hf'.")
    p.add_argument(
        "--n-kv-heads",
        type=str,
        default="auto",
        help=(
            "Target n_kv_heads: int, 'from-template', 'from-hf', or 'auto' (keeps head_group_size constant if possible)."
        ),
    )
    p.add_argument("--d-ff", type=str, default="from-template", help="Target d_ff, or 'from-template'/'from-hf'.")
    p.add_argument("--n-layers", type=str, default="from-hf", help="Target n_layers, or 'from-hf' (default).")
    p.add_argument("--vocab-size", type=str, default="from-hf", help="Target vocab_size, or 'from-hf'.")
    p.add_argument("--theta", type=str, default="from-hf", help="RoPE theta, or 'from-hf'.")

    p.add_argument(
        "--attn-qknorm",
        action="store_true",
        help="Enable qk-norm in the target model (defaults to template value if present, else False).",
    )
    p.add_argument(
        "--qknorm-gain",
        type=float,
        default=None,
        help="If set and --attn-qknorm is enabled, initialize qk-norm gain to this value for all layers.",
    )
    p.add_argument(
        "--attn-val-residual",
        action="store_true",
        help="Enable value residual learning (defaults to template value if present, else False).",
    )
    p.add_argument(
        "--attn-gating",
        type=str,
        default="from-template",
        help="Attention gating: 'false', 'elementwise', 'per-head', 'per-head-hd', or 'from-template'.",
    )
    p.add_argument(
        "--layernorm-scaling",
        action="store_true",
        help="Enable layernorm scaling (defaults to template value if present, else False).",
    )
    p.add_argument(
        "--weight-tying",
        action="store_true",
        help="Tie word embeddings (defaults to template value if present, else HF tie_word_embeddings).",
    )

    args = p.parse_args()

    hf_cfg = AutoConfig.from_pretrained(args.hf_dir)
    hf_hidden = int(getattr(hf_cfg, "hidden_size"))
    hf_heads = int(getattr(hf_cfg, "num_attention_heads"))
    hf_dff = int(getattr(hf_cfg, "intermediate_size"))
    hf_layers = int(getattr(hf_cfg, "num_hidden_layers"))
    hf_vocab = int(getattr(hf_cfg, "vocab_size"))
    hf_theta = float(getattr(hf_cfg, "rope_theta", 10000.0))

    template_cfg: dict[str, Any] | None = None
    if args.template_config is not None:
        import sys

        sys.path.insert(0, str(REPO_ROOT))
        template_cfg = _load_template_config(args.template_config, args.config_key)

    template_model = (template_cfg or {}).get("model", {}) if template_cfg is not None else {}

    width = _parse_from_sources_or_int(args.width, hf_value=hf_hidden, template_value=template_model.get("d_model"))
    n_heads = _parse_from_sources_or_int(args.n_heads, hf_value=hf_heads, template_value=template_model.get("n_heads"))
    d_ff = _parse_from_sources_or_int(args.d_ff, hf_value=hf_dff, template_value=template_model.get("d_ff"))
    n_layers = _parse_from_hf_or_int(args.n_layers, hf_layers) if args.n_layers == "from-hf" else int(args.n_layers)
    vocab_size = (
        _parse_from_hf_or_int(args.vocab_size, hf_vocab) if args.vocab_size == "from-hf" else int(args.vocab_size)
    )
    theta = _parse_from_sources_or_float(args.theta, hf_value=hf_theta, template_value=template_model.get("theta"))

    if width % n_heads != 0:
        raise ValueError(f"Invalid target shape: width={width} not divisible by n_heads={n_heads}")

    attn_qknorm = bool(args.attn_qknorm or template_model.get("attn_qknorm", False))
    attn_val_residual = bool(args.attn_val_residual or template_model.get("attn_val_residual", False))
    layernorm_scaling = bool(args.layernorm_scaling or template_model.get("layernorm_scaling", False))

    if args.attn_gating == "from-template":
        attn_gating: bool | str = template_model.get("attn_gating", False)
    elif args.attn_gating.lower() in {"false", "0", "no"}:
        attn_gating = False
    else:
        attn_gating = args.attn_gating

    if args.weight_tying:
        weight_tying = True
    else:
        weight_tying = bool(template_model.get("weight_tying", getattr(hf_cfg, "tie_word_embeddings", False)))

    hf_kv_heads = int(getattr(hf_cfg, "num_key_value_heads", hf_heads))
    template_kv_heads = template_model.get("n_kv_heads")
    if args.n_kv_heads == "from-hf":
        n_kv_heads = hf_kv_heads
    elif args.n_kv_heads == "from-template":
        n_kv_heads = int(template_kv_heads if template_kv_heads is not None else hf_kv_heads)
    elif args.n_kv_heads == "auto":
        group = hf_heads // hf_kv_heads if (hf_kv_heads > 0 and hf_heads % hf_kv_heads == 0) else 1
        n_kv_heads = n_heads // group if (group > 0 and n_heads % group == 0) else n_heads
    else:
        n_kv_heads = int(args.n_kv_heads)

    _convert(
        hf_dir=args.hf_dir,
        template_cfg=template_cfg,
        output=args.output,
        init_multiplier=args.init_multiplier,
        widening_mode=args.widening_mode,
        width=width,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        vocab_size=vocab_size,
        theta=theta,
        attn_qknorm=attn_qknorm,
        attn_val_residual=attn_val_residual,
        attn_gating=attn_gating,
        layernorm_scaling=layernorm_scaling,
        weight_tying=weight_tying,
        qknorm_gain=args.qknorm_gain,
        ffn_expand=args.ffn_expand,
        ffn_down_noise_std=args.ffn_down_noise_std,
        attn_expand=args.attn_expand,
        attn_out_noise_std=args.attn_out_noise_std,
        expand_seed=args.expand_seed,
    )


if __name__ == "__main__":
    main()
