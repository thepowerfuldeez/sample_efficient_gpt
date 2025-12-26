import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from jaxtyping import Float, Int
from tqdm.auto import tqdm

import torch.distributed as dist

from sample_efficient_gpt.transformer.core import SwiGLU, RMSNorm, Embedding, Linear, softmax
from sample_efficient_gpt.transformer.attention import MultiHeadSelfAttention, KVCache
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
        attn_qknorm: bool = False,
        attn_val_residual: bool = False,
        attn_gating: bool = False,
        n_kv_heads: int | None = None,
        theta: float = 10_000,
        position: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
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
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, position=position, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "b seq d_model"],
        token_positions: Int[Tensor, "b seq"] | None = None,
        v1: Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> Float[Tensor, "b seq d_model"]:
        assert v1 is None or (v1 is not None and (v1.device == x.device))
        attn_out, v = self.attn(self.ln1(x), token_positions, v1=v1, kv_cache=kv_cache)
        # prenorm_act_norm = x.detach().pow(2).mean(dim=-1).sqrt().mean()
        # kurtosis
        x_f = x.detach().to(torch.float32)
        m2 = x_f.pow(2).mean(-1).clamp_min(1e-8)
        m4 = x_f.pow(4).mean(-1)
        avg_kurtosis = (m4 / (m2 * m2)).mean()
        y = x + attn_out
        return y + self.ffn(self.ln2(y)), avg_kurtosis, v


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_qknorm: bool = False,
        attn_val_residual: bool = False,
        attn_gating: bool = False,
        layernorm_scaling: bool = False,
        theta: float = 10_000,
        n_kv_heads: int | None = None,
        device=None,
        dtype=None,
        weight_tying: bool = False,
        num_grad_checkpoint_layers: int = 0,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.n_layers = n_layers
        # do activation checkpointing for first N layers
        self.num_grad_checkpoint_layers = num_grad_checkpoint_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    n_heads,
                    d_ff,
                    n_kv_heads=n_kv_heads,
                    theta=theta,
                    attn_qknorm=attn_qknorm,
                    attn_val_residual=attn_val_residual,
                    attn_gating=attn_gating,
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
            self.lm_head.weight = self.embedding.weight

    def forward(
        self, x: Int[Tensor, "bs seq"], kv_cache: list[KVCache] | None = None
    ) -> Float[Tensor, "bs seq vocab_size"]:
        x: Float[Tensor, "bs seq d_model"] = self.embedding(x)
        avg_kurtosis_values: Float[Tensor, "n_layers"] = torch.zeros(
            (len(self.blocks),), dtype=x.dtype, device=x.device
        )
        v1 = None
        for i, layer in enumerate(self.blocks):
            with nvtx_range(f"block {i}"):
                if i < self.num_grad_checkpoint_layers:
                    # checkpoint this block
                    def layer_forward(x, v1, kv_cache):
                        # IMPORTANT: capture block via closure, pass extra args through
                        return layer(x, v1=v1, kv_cache=kv_cache)

                    # use_reentrant=False plays nicer with torch.compile
                    x, avg_kurtosis, v = checkpoint(
                        layer_forward, x, v1, kv_cache[i] if kv_cache is not None else None, use_reentrant=False
                    )
                else:
                    x, avg_kurtosis, v = layer(x, v1=v1, kv_cache=kv_cache[i] if kv_cache is not None else None)
            if v1 is None:
                v1 = v
            avg_kurtosis_values[i] = avg_kurtosis
        x = self.final_norm(x)
        return self.lm_head(x), avg_kurtosis_values

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
                logits, _ = self.forward(input_seq, kv_cache=kv_cache)
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
