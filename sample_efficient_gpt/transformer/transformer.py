import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from tqdm.auto import tqdm

from sample_efficient_gpt.transformer.core import SwiGLU, RMSNorm, Embedding, Linear, softmax
from sample_efficient_gpt.transformer.attention import MultiHeadSelfAttention
import torch.cuda.nvtx as nvtx
import torch.distributed as dist

_MAX_SEQ_LEN = 4096


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_qknorm: bool = False,
        attn_val_residual: bool = False,
        attn_gating: bool = False,
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
            theta,
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
    ) -> Float[Tensor, "b seq d_model"]:
        assert v1 is None or (v1 is not None and (v1.device == x.device))
        attn_out, v = self.attn(self.ln1(x), token_positions, v1=v1)
        prenorm_act_norm = x.detach().pow(2).mean(dim=-1).sqrt().mean()
        y = x + attn_out
        return y + self.ffn(self.ln2(y)), prenorm_act_norm, v

    def forward_tp(
        self,
        x: Float[Tensor, "b seq d_model / 2"],
        token_positions: Int[Tensor, "b seq"] | None = None,
        v1: Tensor | None = None,
    ) -> Float[Tensor, "b seq d_model"]:
        assert dist.is_initialized()

        x: Float[Tensor, "b seq d_model / 2"] = self.ln1(x)
        activations: list = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=activations, tensor=x)
        x: Float[Tensor, "b seq d_model"] = torch.cat(x, dim=-1)

        attn_out, v = self.attn(x, token_positions, v1=v1)
        # attn_out: Float["bs seq d_model / 2"]
        dist.reduce_scatter(attn_out)

        # prenorm_act_norm = x.detach().pow(2).mean(dim=-1).sqrt().mean()
        y = x + attn_out
        return y + self.ffn(self.ln2(y)), v


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
        device=None,
        dtype=None,
        weight_tying: bool = False,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    n_heads,
                    d_ff,
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

    def forward(self, x: Int[Tensor, "bs seq"]) -> Float[Tensor, "bs seq vocab_size"]:
        x: Float[Tensor, "bs seq d_model"] = self.embedding(x)
        prenorm_activation_norms: Float[Tensor, "n_layers"] = torch.zeros(
            (len(self.blocks),), dtype=x.dtype, device=x.device
        )
        v1 = None
        for i, layer in enumerate(self.blocks):
            # with nvtx.range(f"block {i}"):
            # pass residual value
            x, prenorm_act_norm, v = layer(x, v1=v1)
            if v1 is None:
                v1 = v
            prenorm_activation_norms[i] = prenorm_act_norm
        x = self.final_norm(x)
        return self.lm_head(x), prenorm_activation_norms

    def generate(
        self,
        prompt: Int[Tensor, "bs seq"],
        eos_token_id: int,
        top_p: float = 0.4,
        temperature: float = 0.7,
        max_steps: int = 32,
    ):
        """
        Perform decoding with nucleous sampling and temperature
        """
        input_seq = prompt
        with torch.inference_mode():
            for _ in tqdm(range(max_steps)):
                logits: Float[Tensor, "bs seq vocab"]
                logits, _ = self.forward(input_seq)
                if temperature == 0:
                    out: Int[Tensor, "bs"] = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                else:
                    probs: Float[Tensor, "bs vocab"] = softmax(logits, dim=-1, temperature=temperature)[:, -1, :]
                    # nucleous sampling
                    if top_p < 1.0:
                        sorted_values, sorted_idx = probs.sort(-1, descending=True)
                        mask = sorted_values.cumsum(-1) <= top_p
                        mask[:, 0] = True
                        new_probs_mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, sorted_idx, mask)
                        probs = probs * new_probs_mask
                        probs = probs / probs.sum(-1, keepdim=True)
                    out: Int[Tensor, "bs"] = torch.multinomial(probs, 1)
                input_seq = torch.cat([input_seq, out], dim=-1)
                if (out[-1:] == eos_token_id).all(dim=-1).item():
                    break
        return input_seq

    def forward_tp(self, x: Int[Tensor, "bs seq"]) -> Float[Tensor, "bs seq vocab_size"]:
        x: Float[Tensor, "bs seq d_model"] = self.embedding(x)
        prenorm_activation_norms: Float[Tensor, "n_layers"] = torch.zeros(
            (len(self.blocks),), dtype=x.dtype, device=x.device
        )
        v1 = None
        for i, layer in enumerate(self.blocks):
            # pass residual value
            with nvtx.range(f"block {i}"):
                x, prenorm_act_norm, v = layer(x, v1=v1)
                if v1 is None:
                    v1 = v
                prenorm_activation_norms[i] = prenorm_act_norm
        x = self.final_norm(x)
        return self.lm_head(x), prenorm_activation_norms


if __name__ == "__main__":
    d_model = 1024
    d_ff = 2048
    num_heads = 8
    theta = 10_000
    block = Block(d_model, num_heads, theta, d_ff / d_model)
    x = torch.randn(4, 64, 1024)

    print(block(x).shape)
