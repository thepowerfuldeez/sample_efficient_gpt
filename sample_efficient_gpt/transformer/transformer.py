import os
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from jaxtyping import Float, Int
from tqdm.auto import tqdm

import torch.distributed as dist

from sample_efficient_gpt.transformer.core import SwiGLU, RMSNorm, Embedding, Linear, softmax
from sample_efficient_gpt.transformer.attention import MultiHeadSelfAttention, KVCache
from sample_efficient_gpt.transformer.moe import TopKMoE
from sample_efficient_gpt.utils.profiling import nvtx_range

_MAX_SEQ_LEN = 8192


class CheckpointBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x, *args, **kwargs):
        def _forward(*inputs):
            # re-pack if your block takes more than just x
            return self.block(*inputs, **kwargs)

        return checkpoint(_forward, x, *args)


def apply_checkpointing_to_last_n_layers(model, n_last: int):
    layers = model.transformer.blocks  # adapt to your naming
    L = len(layers)
    for i in range(L - n_last, L):
        layers[i] = CheckpointBlock(layers[i])


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        use_moe: bool = False,
        moe_backend: str = "native",
        moe_num_experts: int = 0,
        moe_top_k: int = 1,
        moe_capacity_factor: float = 1.0,
        moe_aux_loss_coef: float = 0.01,
        moe_z_loss_coef: float = 0.0,
        moe_router_jitter: float = 0.0,
        moe_normalize_gates: bool = True,
        moe_gate_scale: float = 1.0,
        moe_expert_parallel_size: int = 1,
        moe_expert_precision: str = "bf16",
        attn_qknorm: bool = False,
        attn_val_residual: bool = False,
        attn_gating: bool = False,
        theta: float = 10_000,
        rope_interleaved: bool = False,
        n_kv_heads: int | None = None,
        position: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.track_kurtosis = os.environ.get("SEGPT_TRACK_KURTOSIS", "1") == "1"
        self.ln1 = RMSNorm(d_model, position=position, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model,
            n_heads,
            n_kv_heads=n_kv_heads,
            theta=theta,
            max_seq_len=_MAX_SEQ_LEN,
            qknorm=attn_qknorm,
            value_residual=attn_val_residual,
            gating=attn_gating,
            rope_interleaved=rope_interleaved,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, position=position, device=device, dtype=dtype)
        if use_moe and moe_num_experts > 0:
            if moe_backend == "native":
                self.ffn = TopKMoE(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_experts=moe_num_experts,
                    top_k=moe_top_k,
                    capacity_factor=moe_capacity_factor,
                    aux_loss_coef=moe_aux_loss_coef,
                    z_loss_coef=moe_z_loss_coef,
                    router_jitter=moe_router_jitter,
                    normalize_gates=moe_normalize_gates,
                    gate_scale=moe_gate_scale,
                    expert_parallel_size=moe_expert_parallel_size,
                    expert_precision=moe_expert_precision,
                    device=device,
                    dtype=dtype,
                )
            elif moe_backend == "sonicmoe":
                if moe_expert_parallel_size != 1:
                    raise ValueError("moe_backend='sonicmoe' currently does not support expert parallelism.")
                if moe_expert_precision.lower() != "bf16":
                    raise ValueError("moe_backend='sonicmoe' currently requires moe_expert_precision='bf16'.")
                if moe_z_loss_coef != 0.0:
                    raise ValueError("moe_backend='sonicmoe' does not support moe_z_loss_coef (set it to 0).")
                if moe_capacity_factor != 1.0:
                    raise ValueError("moe_backend='sonicmoe' does not support moe_capacity_factor (set it to 1).")
                if not moe_normalize_gates:
                    raise ValueError("moe_backend='sonicmoe' does not support moe_normalize_gates=False.")
                if moe_gate_scale != 1.0:
                    raise ValueError("moe_backend='sonicmoe' does not support moe_gate_scale (set it to 1).")

                from sample_efficient_gpt.transformer.sonic_moe import SonicMoEAdapter

                self.ffn = SonicMoEAdapter(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_experts=moe_num_experts,
                    top_k=moe_top_k,
                    aux_loss_coef=moe_aux_loss_coef,
                    device=device,
                    dtype=dtype,
                )
            else:
                raise ValueError(f"Unknown moe_backend: {moe_backend!r}")
            self._ffn_is_moe = True
        else:
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
            self._ffn_is_moe = False

    def forward(
        self,
        x: Float[Tensor, "b seq d_model"],
        token_positions: Int[Tensor, "b seq"] | None = None,
        v1: Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> Float[Tensor, "b seq d_model"]:
        assert v1 is None or (v1 is not None and (v1.device == x.device))
        attn_out, v = self.attn(self.ln1(x), token_positions, v1=v1, kv_cache=kv_cache)
        if self.track_kurtosis:
            x_f = x.detach().to(torch.float32)
            m2 = x_f.pow(2).mean(-1).clamp_min(1e-8)
            m4 = x_f.pow(4).mean(-1)
            avg_kurtosis = (m4 / (m2 * m2)).mean()
        else:
            avg_kurtosis = torch.zeros((), device=x.device, dtype=x.dtype)
        y = x + attn_out
        if self._ffn_is_moe:
            ffn_out, moe_aux = self.ffn(self.ln2(y))
        else:
            ffn_out = self.ffn(self.ln2(y))
            moe_aux = torch.zeros((), device=y.device, dtype=y.dtype)
        return y + ffn_out, avg_kurtosis, v, moe_aux


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_kv_heads: int | None = None,
        attn_qknorm: bool = False,
        attn_val_residual: bool = False,
        attn_gating: bool = False,
        layernorm_scaling: bool = False,
        theta: float = 10_000,
        rope_interleaved: bool = False,
        device=None,
        dtype=None,
        weight_tying: bool = False,
        num_grad_checkpoint_layers: int = 15,
        moe_backend: str = "native",
        moe_num_experts: int = 0,
        moe_top_k: int = 1,
        moe_capacity_factor: float = 1.0,
        moe_aux_loss_coef: float = 0.01,
        moe_z_loss_coef: float = 0.0,
        moe_router_jitter: float = 0.0,
        moe_normalize_gates: bool = True,
        moe_gate_scale: float = 1.0,
        moe_expert_parallel_size: int = 1,
        moe_expert_precision: str = "bf16",
        moe_start_layer: int = 0,
        moe_every_n_layers: int = 1,
        moe_end_layer: int | None = None,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.n_layers = n_layers
        # do activation checkpointing for first N layers
        self.num_grad_checkpoint_layers = num_grad_checkpoint_layers
        self.moe_backend = str(moe_backend)
        self.moe_num_experts = int(moe_num_experts)
        self.moe_top_k = int(moe_top_k)
        self.moe_capacity_factor = float(moe_capacity_factor)
        self.moe_aux_loss_coef = float(moe_aux_loss_coef)
        self.moe_z_loss_coef = float(moe_z_loss_coef)
        self.moe_router_jitter = float(moe_router_jitter)
        self.moe_normalize_gates = bool(moe_normalize_gates)
        self.moe_gate_scale = float(moe_gate_scale)
        self.moe_expert_parallel_size = int(moe_expert_parallel_size)
        self.moe_expert_precision = str(moe_expert_precision)
        self.moe_start_layer = int(moe_start_layer)
        self.moe_every_n_layers = int(moe_every_n_layers)
        self.moe_end_layer = int(moe_end_layer) if moe_end_layer is not None else None

        def use_moe_for_layer(layer_idx0: int) -> bool:
            if self.moe_num_experts <= 0:
                return False
            if layer_idx0 < self.moe_start_layer:
                return False
            if self.moe_end_layer is not None and layer_idx0 >= self.moe_end_layer:
                return False
            if self.moe_every_n_layers <= 0:
                return False
            return ((layer_idx0 - self.moe_start_layer) % self.moe_every_n_layers) == 0

        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    n_heads,
                    d_ff,
                    use_moe=use_moe_for_layer(pos - 1),
                    moe_backend=self.moe_backend,
                    moe_num_experts=self.moe_num_experts,
                    moe_top_k=self.moe_top_k,
                    moe_capacity_factor=self.moe_capacity_factor,
                    moe_aux_loss_coef=self.moe_aux_loss_coef,
                    moe_z_loss_coef=self.moe_z_loss_coef,
                    moe_router_jitter=self.moe_router_jitter,
                    moe_normalize_gates=self.moe_normalize_gates,
                    moe_gate_scale=self.moe_gate_scale,
                    moe_expert_parallel_size=self.moe_expert_parallel_size,
                    moe_expert_precision=self.moe_expert_precision,
                    theta=theta,
                    attn_qknorm=attn_qknorm,
                    attn_val_residual=attn_val_residual,
                    attn_gating=attn_gating,
                    rope_interleaved=rope_interleaved,
                    n_kv_heads=n_kv_heads,
                    position=pos if layernorm_scaling else None,
                    device=device,
                    dtype=dtype,
                )
                for pos in range(1, n_layers + 1)
            ]
        )
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
        if weight_tying:
            # Tie the actual Linear weight used in forward().
            self.lm_head.linear.weight = self.embedding.weight

    def forward(
        self, x: Int[Tensor, "bs seq"], kv_cache: list[KVCache] | None = None
    ) -> tuple[Float[Tensor, "bs seq vocab_size"], Float[Tensor, "n_layers"], Tensor]:
        x: Float[Tensor, "bs seq d_model"] = self.embedding(x)
        avg_kurtosis_values: Float[Tensor, "n_layers"] = torch.zeros(
            (len(self.blocks),), dtype=x.dtype, device=x.device
        )
        moe_aux_loss = torch.zeros((), dtype=x.dtype, device=x.device)
        v1 = None
        for i, layer in enumerate(self.blocks):
            with nvtx_range(f"block {i}"):
                # EP MoE uses all-to-all collectives in forward; avoid activation checkpointing for those blocks
                # (checkpoint would re-run collectives during backward, increasing overhead and risking hangs).
                is_ep_moe = bool(
                    getattr(layer, "_ffn_is_moe", False)
                    and isinstance(getattr(layer, "ffn", None), TopKMoE)
                    and getattr(layer.ffn, "_ep_enabled", False)
                )
                if i < self.num_grad_checkpoint_layers and not is_ep_moe:
                    # checkpoint this block
                    def layer_forward(x, v1, kv_cache, _layer=layer):
                        # IMPORTANT: bind `layer` as a default arg to avoid late-binding bugs during recomputation.
                        return _layer(x, v1=v1, kv_cache=kv_cache)

                    # use_reentrant=False plays nicer with torch.compile
                    x, avg_kurtosis, v, moe_aux = checkpoint(
                        layer_forward, x, v1, kv_cache[i] if kv_cache is not None else None, use_reentrant=False
                    )
                else:
                    x, avg_kurtosis, v, moe_aux = layer(
                        x, v1=v1, kv_cache=kv_cache[i] if kv_cache is not None else None
                    )
            if v1 is None:
                v1 = v
            avg_kurtosis_values[i] = avg_kurtosis
            moe_aux_loss = moe_aux_loss + moe_aux
        x = self.final_norm(x)
        return self.lm_head(x), avg_kurtosis_values, moe_aux_loss

    def maybe_convert_moe_experts_to_fp8(self) -> None:
        """
        Convert only MoE expert matmuls to float8 (best-effort).

        Call after moving the model to CUDA and before initializing the optimizer / DDP buckets.
        """
        if self.moe_num_experts <= 0:
            return
        if self.moe_backend != "native":
            return
        if self.moe_expert_precision.lower() != "fp8":
            return
        for m in self.modules():
            if isinstance(m, TopKMoE):
                m.convert_experts_to_fp8()

    def generate(
        self,
        prompt: Int[Tensor, "bs seq"],
        eos_token_id: int,
        top_p: float = 0.4,
        temperature: float = 0.7,
        max_steps: int = 32,
        eos_prob_multiplier: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform decoding with nucleous sampling and temperature

        returns generated sampled token ids + EOS probs at each step for debugging
        """
        kv_cache = [KVCache() for _ in range(self.n_layers)]
        input_seq = prompt.clone()
        output_seq = prompt
        eos_probs = None
        with torch.inference_mode():
            for _ in tqdm(range(max_steps)):
                logits: Float[Tensor, "bs seq vocab"]
                logits, *_ = self.forward(input_seq, kv_cache=kv_cache)
                all_probs = softmax(logits, dim=-1, temperature=temperature)
                last_prob: Float[Tensor, "bs vocab"] = all_probs[:, -1, :]
                if temperature == 0:
                    out: Int[Tensor, "bs"] = torch.argmax(last_prob, dim=-1, keepdim=True)
                else:
                    last_prob[:, eos_token_id] *= eos_prob_multiplier
                    # nucleous sampling
                    if top_p < 1.0:
                        sorted_values, sorted_idx = last_prob.sort(-1, descending=True)
                        mask = sorted_values.cumsum(-1) <= top_p
                        mask[:, 0] = True
                        new_probs_mask = torch.zeros_like(last_prob, dtype=torch.bool).scatter_(-1, sorted_idx, mask)
                        last_prob = last_prob * new_probs_mask
                        last_prob = last_prob / last_prob.sum(-1, keepdim=True)
                    out: Int[Tensor, "bs"] = torch.multinomial(last_prob, 1)
                output_seq = torch.cat([output_seq, out], dim=-1)
                input_seq = out
                # input_seq = output_seq
                last_prob_eos = last_prob[:, eos_token_id]
                if eos_probs is None:
                    eos_probs = last_prob_eos
                else:
                    eos_probs = torch.cat((eos_probs, last_prob_eos), dim=-1)
                if (out == eos_token_id).all(dim=-1).item():
                    break
        return output_seq, eos_probs


if __name__ == "__main__":
    d_model = 1024
    d_ff = 2048
    num_heads = 8
    theta = 10_000
    block = Block(d_model, num_heads, theta, d_ff / d_model)
    x = torch.randn(4, 64, 1024)

    print(block(x).shape)
