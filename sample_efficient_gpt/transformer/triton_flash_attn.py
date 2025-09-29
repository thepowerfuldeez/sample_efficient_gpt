import torch
from torch import Tensor
from jaxtyping import Float

import triton
import triton.language as tl

torch.cuda.set_device("cuda:0")
torch.set_float32_matmul_precision("high")


# fmt: off
@triton.jit
def flashatt_kernel_fwd(q_ptr, k_ptr, v_ptr, z_ptr, l_ptr, 
                        Nq, Nk, scale,
                        D: tl.constexpr, 
                        Q_TILE_SIZE: tl.constexpr, 
                        K_TILE_SIZE: tl.constexpr, 
                        IS_CAUSAL: tl.constexpr):
    log2_e = 1.44269504
    # query loop index
    i = tl.program_id(0)
    # batch index
    b = tl.program_id(1)

    q_idx = tl.arange(0, Q_TILE_SIZE)
    k_idx = tl.arange(0, K_TILE_SIZE)

    # strides (Nq * d, d, 1)
    q_bp = tl.make_block_ptr(
        q_ptr + b * Nq * D, (Nq, D), (D, 1), (i * Q_TILE_SIZE, 0), (Q_TILE_SIZE, D), (1, 0)
    )
    k_bp = tl.make_block_ptr(
        k_ptr + b * Nk * D, (Nk, D), (D, 1), (0, 0), (K_TILE_SIZE, D), (1, 0)
    )
    v_bp = tl.make_block_ptr(
        v_ptr + b * Nk * D, (Nk, D), (D, 1), (0, 0), (K_TILE_SIZE, D), (1, 0)
    )
    z_bp = tl.make_block_ptr(
        z_ptr + b * Nq * D, (Nq, D), (D, 1), (i * Q_TILE_SIZE, 0), (Q_TILE_SIZE, D), (1, 0)
    )
    l_bp = tl.make_block_ptr(
        l_ptr + b * Nq, (Nq, ), (1, ), (i * Q_TILE_SIZE, ), (Q_TILE_SIZE, ), (0, )
    )

    q_i = tl.load(q_bp, boundary_check=(0, 1))
    out = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE, 1), float("-inf"), dtype=tl.float32)
    last_m = m

    # Nk_adj = (i * Q_TILE_SIZE / Nq * Nk) // K_TILE_SIZE * K_TILE_SIZE + 1
    global_q_idx = q_idx + i * Q_TILE_SIZE

    # if j > i * Q_TILE_SIZE: skip that block
    if IS_CAUSAL:
        end_idx = i * Q_TILE_SIZE + K_TILE_SIZE
    else:
        end_idx = Nk

    q_i = q_i.to(tl.float16)
    for j in range(0, end_idx, K_TILE_SIZE):
        k_j, v_j = tl.load(k_bp, boundary_check=(0, 1)), tl.load(v_bp, boundary_check=(0, 1))

        k_j = k_j.to(tl.float16)
        v_j = v_j.to(tl.float16)

        s = tl.dot(q_i, k_j.T) * scale
        m = tl.maximum(last_m, s.max(axis=-1, keep_dims=True))

        if IS_CAUSAL:
            # we only care about single diagonal block, all preceding are full blocks (no need masking)
            if j >= i * Q_TILE_SIZE:
                global_k_idx = k_idx + j
                # apply causal mask
                s = tl.where(global_q_idx[:, None] >= global_k_idx[None, :], s, float("-inf"))

        p = tl.exp2(log2_e * (s - m))
        l = tl.exp2(log2_e * (last_m - m)) * l + p.sum(axis=-1, keep_dims=True)

        p = p.to(v_j.dtype)
        out = tl.exp2(log2_e * (last_m - m)) * out + tl.dot(p, v_j)
        last_m = m

        k_bp = k_bp.advance((K_TILE_SIZE, 0))
        v_bp = v_bp.advance((K_TILE_SIZE, 0))
    # Write to a single O / L block
    out = out / l
    l = m + tl.log(l)
    l = l.reshape((Q_TILE_SIZE, ))
    tl.store(z_bp, out.to(z_bp.dtype.element_ty), boundary_check=(0, 1))
    tl.store(l_bp, l.to(l_bp.dtype.element_ty), boundary_check=(0, ))
# fmt: on


class TritonFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "*Nq d"],
        K: Float[Tensor, "*Nk d"],
        V: Float[Tensor, "*Nk d"],
        is_causal: bool = True,
        q_tile: int = 32,
        k_tile: int = 32,
    ):
        assert Q.is_cuda and Q.is_contiguous()
        # need to add batch dim
        orig_shape = Q.shape
        if Q.ndim == 2:
            Q = Q.unsqueeze(0)
            K = K.unsqueeze(0)
            V = V.unsqueeze(0)

        bs, Nq, d = Q.shape
        Nk = K.shape[1]
        scale = 1 / (d**0.5)

        out_final = torch.empty(bs, Nq, d, device=Q.device, dtype=Q.dtype)
        l_final = torch.empty(bs, Nq, device=Q.device, dtype=Q.dtype)

        grid = lambda meta: (triton.cdiv(Nq, meta["Q_TILE_SIZE"]), bs)

        flashatt_kernel_fwd[grid](
            Q,
            K,
            V,
            out_final,
            l_final,
            Nq,
            Nk,
            scale,
            d,
            q_tile,
            k_tile,
            is_causal,
        )
        ctx.is_causal = is_causal
        ctx.save_for_backward(l_final, Q, K, V, out_final)
        return out_final.view(orig_shape)

    @torch.compile()
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors
        bs, Nq, d = Q.shape
        Nk = K.shape[1]
        scale = 1 / (d**0.5)
        D = (O * grad_output).sum(-1, keepdim=True)

        S = Q @ K.transpose(1, 2) * scale
        if ctx.is_causal:
            m = torch.full((1, Nq, Nk), True, dtype=torch.bool, device=Q.device)
            m = torch.tril(m)
            S = torch.where(m, S, float("-inf"))

        # recomputation
        P = (S - L[..., None]).exp()

        dV = P.transpose(1, 2) @ grad_output
        dP = grad_output @ V.transpose(1, 2)
        dS = P * (dP - D)
        dQ = dS @ K * scale
        dK = dS.transpose(1, 2) @ Q * scale
        return dQ, dK, dV, None, None, None


if __name__ == "__main__":
    Q, K, V = (
        torch.randn(256, 64, dtype=torch.float32, device="cuda"),
        torch.randn(128, 64, dtype=torch.float32, device="cuda"),
        torch.randn(128, 64, dtype=torch.float32, device="cuda"),
    )

    out = TritonFlashAttnFunc.apply(Q, K, V)

    print(out.shape)
    print(out)

    out2 = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    print(out2)

    # print((out - out2).abs().mean())
    # print(torch.allclose(out, out2, atol=1e-6))
