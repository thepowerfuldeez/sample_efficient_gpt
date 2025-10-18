import torch
from torch import Tensor
from jaxtyping import Float

import torch._dynamo as dynamo
import triton
import triton.language as tl

torch.cuda.set_device("cuda:0")
torch.set_float32_matmul_precision("high")


# fmt: off
@triton.autotune(configs=[
    triton.Config(kwargs={'Q_TILE_SIZE': 32, "K_TILE_SIZE": 32}, num_warps=8),
    triton.Config(kwargs={'Q_TILE_SIZE': 64, "K_TILE_SIZE": 64}, num_warps=4),
    triton.Config(kwargs={'Q_TILE_SIZE': 64, "K_TILE_SIZE": 64}, num_warps=2),
    triton.Config(kwargs={'Q_TILE_SIZE': 128, "K_TILE_SIZE": 128}, num_warps=2),
  ],
  key=['Nq'] # the two above configs will be evaluated anytime
                 # the value of x_size changes
)
@triton.jit
def flashatt_qknorm_kernel_fwd(q_ptr, k_ptr, v_ptr, scale_ptr, 
                        z_ptr, l_ptr, 
                        Nq, Nk,
                        q_start,
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

    global_q_idx = q_idx + i * Q_TILE_SIZE + q_start

    # q-norm
    qf = q_i.to(tl.float32)
    alpha = tl.rsqrt((qf * qf).sum(axis=1, keep_dims=True) + 1e-8)

    # if j > i * Q_TILE_SIZE: skip that block
    if IS_CAUSAL:
        if Q_TILE_SIZE != K_TILE_SIZE:
            end_idx = tl.minimum(Nk, q_start + (i + 1) * Q_TILE_SIZE)
        else:
            end_idx = i * Q_TILE_SIZE + K_TILE_SIZE + q_start
    else:
        end_idx = Nk

    scale = tl.load(scale_ptr).to(tl.bfloat16)

    q_i = q_i.to(tl.bfloat16)
    for j in range(0, end_idx, K_TILE_SIZE):
        k_j, v_j = tl.load(k_bp, boundary_check=(0, 1)), tl.load(v_bp, boundary_check=(0, 1))

        # k-norm
        kf = k_j.to(tl.float32)
        beta = tl.rsqrt((kf * kf).sum(axis=1) + 1e-8)[None, :]

        k_j = k_j.to(tl.bfloat16)
        v_j = v_j.to(tl.bfloat16)

        s = tl.dot(q_i, k_j.T) * alpha * beta * scale
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


@triton.autotune(configs=[
    triton.Config(kwargs={'Q_TILE_SIZE': 32, "K_TILE_SIZE": 32}),
    triton.Config(kwargs={'Q_TILE_SIZE': 64, "K_TILE_SIZE": 64}),
  ],
  key=['Nq'] # the two above configs will be evaluated anytime
                 # the value of x_size changes
)
@triton.jit()
def flashatt_qknorm_kernel_bwd(
                        q_ptr, k_ptr, v_ptr, scale_ptr, z_ptr, l_ptr, 
                        grad_out_ptr,
                        dq_ptr, dk_ptr, dv_ptr, dg_ptr,
                        Nq, Nk,
                        D: tl.constexpr, 
                        Q_TILE_SIZE: tl.constexpr, 
                        K_TILE_SIZE: tl.constexpr, 
                        IS_CAUSAL: tl.constexpr
    ):
    log2_e = 1.44269504
    # key loop index
    j = tl.program_id(0)
    # batch index
    b = tl.program_id(1)

    q_idx = tl.arange(0, Q_TILE_SIZE)
    k_idx = tl.arange(0, K_TILE_SIZE)

    # strides (Nq * d, d, 1)
    q_bp = tl.make_block_ptr(
        q_ptr + b * Nq * D, (Nq, D), (D, 1), (0, 0), (Q_TILE_SIZE, D), (1, 0)
    )
    z_bp = tl.make_block_ptr(
        z_ptr + b * Nq * D, (Nq, D), (D, 1), (0, 0), (Q_TILE_SIZE, D), (1, 0)
    )
    grad_out_bp = tl.make_block_ptr(
        grad_out_ptr + b * Nq * D, (Nq, D), (D, 1), (0, 0), (Q_TILE_SIZE, D), (1, 0)
    )
    dq_bp = tl.make_block_ptr(
        dq_ptr + b * Nq * D, (Nq, D), (D, 1), (0, 0), (Q_TILE_SIZE, D), (1, 0)
    )

    k_bp = tl.make_block_ptr(
        k_ptr + b * Nk * D, (Nk, D), (D, 1), (j * K_TILE_SIZE, 0), (K_TILE_SIZE, D), (1, 0)
    )
    v_bp = tl.make_block_ptr(
        v_ptr + b * Nk * D, (Nk, D), (D, 1), (j * K_TILE_SIZE, 0), (K_TILE_SIZE, D), (1, 0)
    )
    dk_bp = tl.make_block_ptr(
        dk_ptr + b * Nk * D, (Nk, D), (D, 1), (j * K_TILE_SIZE, 0), (K_TILE_SIZE, D), (1, 0)
    )
    dv_bp = tl.make_block_ptr(
        dv_ptr + b * Nk * D, (Nk, D), (D, 1), (j * K_TILE_SIZE, 0), (K_TILE_SIZE, D), (1, 0)
    )
    
    l_bp = tl.make_block_ptr(
        l_ptr + b * Nq, (Nq, ), (1, ), (0, ), (Q_TILE_SIZE, ), (0, )
    )

    # (Bk, D)
    k_j, v_j = tl.load(k_bp, boundary_check=(0, 1)), tl.load(v_bp, boundary_check=(0, 1))
    k_f = k_j.to(tl.float32)
    v_j = v_j.to(tl.bfloat16)

    # k-norm
    beta = tl.rsqrt((k_f * k_f).sum(axis=1) + 1e-8)[None, :]
    scale = tl.load(scale_ptr)

    dk, dv = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32), tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dg = tl.zeros((), dtype=tl.float32)

    q_idx = tl.arange(0, Q_TILE_SIZE)
    k_idx = tl.arange(0, K_TILE_SIZE)
    global_k_idx = k_idx + j * K_TILE_SIZE

    for i in range(0, Nq, Q_TILE_SIZE):
        if IS_CAUSAL and i >= j * K_TILE_SIZE or not IS_CAUSAL:
            q_i = tl.load(q_bp, boundary_check=(0, 1))
            q_f = q_i.to(tl.float32)
            o_i = tl.load(z_bp, boundary_check=(0, 1)).to(tl.bfloat16)
            grad_out_i = tl.load(grad_out_bp, boundary_check=(0, 1)).to(tl.bfloat16)

            # q-norm
            alpha = tl.rsqrt((q_f * q_f).sum(axis=1, keep_dims=True) + 1e-8)

            d = (grad_out_i * o_i).sum(-1, keep_dims=True)
            L_i = tl.load(l_bp, boundary_check=(0, ))[:, None]

            # keep raw dot in f32
            s = tl.dot(q_i, k_j.T).to(tl.float32) * alpha * beta * scale                     # (Bq, Bk)
            if IS_CAUSAL:
                if i <= j * K_TILE_SIZE:
                    global_q_idx = q_idx + i
                    # apply causal mask
                    s = tl.where(global_q_idx[:, None] >= global_k_idx[None, :], s, float("-inf"))
            
            p = tl.exp2(log2_e * (s - L_i)).to(grad_out_i.dtype)
            dv += tl.dot(p.T, grad_out_i)
            dp = tl.dot(grad_out_i, v_j.T)
            ds = (p.to(tl.float32)) * ((dp - d).to(tl.float32))

            # dQ tile
            t1_Q   = tl.dot(ds * beta, k_f)                  # (Bq, D)
            # t2_Q   = tl.sum((ds * beta) * r, axis=1)         # (Bq,)
            t2_Q = tl.sum(t1_Q * q_f, axis=1)

            alpha3 = alpha * alpha * alpha                   # (Bq,1)
            dq_tile = scale * (alpha * t1_Q - alpha3 * (t2_Q[:, None] * q_f))

            cols = i + q_idx
            mask = cols[:, None] < Nq
            dq_range = dq_ptr + b * Nq * D + (cols[:, None] * D) + tl.arange(0, D)[None, :]
            tl.atomic_add(dq_range, dq_tile, mask=mask)

            # dK tile
            t1_K   = tl.dot(tl.trans(ds * alpha), q_f)       # (Bk, D)
            # t2_K   = tl.sum((ds * alpha) * r, axis=0)        # (Bk,)
            t2_K = tl.sum(t1_K * k_f, axis=1)

            beta_col = beta.T                                 # (Bk,1)
            beta3    = beta_col * beta_col * beta_col
            dk += scale * (beta_col * t1_K - beta3 * (t2_K[:, None] * k_f))

            # # center scores row-wise (same shift as softmax)
            # s_center = (s - L_i).to(tl.float32)
            # # kill masked / underflowed positions to avoid 0 * (-inf) -> NaN
            # s_center = tl.where(p > 0, s_center, 0.0)
            # dg = (ds * s_center / (scale + 1e-10)).sum()

            dg += tl.sum(alpha * tl.sum(q_f * t1_Q, axis=1))

        q_bp = q_bp.advance((Q_TILE_SIZE, 0))
        z_bp = z_bp.advance((Q_TILE_SIZE, 0))
        dq_bp = dq_bp.advance((Q_TILE_SIZE, 0))
        grad_out_bp = grad_out_bp.advance((Q_TILE_SIZE, 0))
        l_bp = l_bp.advance((Q_TILE_SIZE, ))
    tl.atomic_add(dg_ptr, dg.to(scale.dtype))
    tl.store(dk_bp, dk.to(dk_bp.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dv_bp, dv.to(dv_bp.dtype.element_ty), boundary_check=(0, 1))
# fmt: on


# Optimal is q_tile / k_tile is (128, 64) for fwd and (32, 64) for bwd


class TritonFlashAttnQKNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "*Nq d"],
        K: Float[Tensor, "*Nk d"],
        V: Float[Tensor, "*Nk d"],
        gain: Float[Tensor, "1"],
        is_causal: bool = True,
        q_start: int = 0,
        q_tile: int = 64,
        k_tile: int = 64,
    ):
        assert Q.is_cuda and Q.is_contiguous()
        assert K.is_cuda and K.is_contiguous()
        assert V.is_cuda and V.is_contiguous()
        assert gain.is_cuda
        # need to add batch dim
        if Q.ndim == 2:
            Q = Q.unsqueeze(0)
            K = K.unsqueeze(0)
            V = V.unsqueeze(0)

        bs, Nq, d = Q.shape
        Nk = K.shape[1]

        out_final = torch.empty_like(Q)
        l_final = torch.empty(bs, Nq, device=Q.device, dtype=Q.dtype)
        grid = lambda meta: (triton.cdiv(Nq, meta["Q_TILE_SIZE"]), bs)

        flashatt_qknorm_kernel_fwd[grid](
            Q,
            K,
            V,
            gain,
            out_final,
            l_final,
            Nq,
            Nk,
            q_start,
            d,
            # q_tile,
            # k_tile,
            IS_CAUSAL=is_causal,
        )
        ctx.is_causal = is_causal
        ctx.save_for_backward(l_final, Q, K, V, gain, out_final)
        return out_final

    def backward(ctx, grad_output, q_tile: int = 64, k_tile: int = 64):
        L, Q, K, V, gain, O = ctx.saved_tensors
        bs, Nq, d = Q.shape
        Nk = K.shape[1]

        def maybe_contiguous(x):
            assert x.is_cuda
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        grad_output, Q, K, V, O = [maybe_contiguous(x) for x in [grad_output, Q, K, V, O]]

        dQ = torch.zeros_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        dg = torch.zeros_like(gain)

        grid = lambda meta: (triton.cdiv(Nk, meta["K_TILE_SIZE"]), bs)
        flashatt_qknorm_kernel_bwd[grid](
            Q,
            K,
            V,
            gain,
            O,
            L,
            grad_output,
            dQ,
            dK,
            dV,
            dg,
            Nq,
            Nk,
            D=d,
            # Q_TILE_SIZE=q_tile,
            # K_TILE_SIZE=k_tile,
            IS_CAUSAL=ctx.is_causal,
        )
        return dQ, dK, dV, dg, None, None, None


if __name__ == "__main__":
    Q, K, V = (
        torch.randn(1, 256, 64, dtype=torch.float32, device="cuda", requires_grad=True),
        torch.randn(1, 256, 64, dtype=torch.float32, device="cuda", requires_grad=True),
        torch.randn(1, 256, 64, dtype=torch.float32, device="cuda", requires_grad=True),
    )
    from sample_efficient_gpt.transformer.attention import SelfDotProductAttnQKNorm

    sdpa = SelfDotProductAttnQKNorm(64)
    sdpa = sdpa.to("cuda")
    gamma = sdpa.gain

    out = TritonFlashAttnQKNormFunc.apply(Q, K, V, gamma)

    print(out.shape)
    print(out)
    out.sum().backward()

    dq, dk, dv, dg = Q.grad, K.grad, V.grad, gamma.grad

    Q.grad = None
    K.grad = None
    V.grad = None
    gamma.grad = None

    seq_len = Q.shape[1]
    out2 = sdpa(Q, K, V, True)
    print(out2)

    out2.sum().backward()

    dq2, dk2, dv2, dg2 = Q.grad, K.grad, V.grad, gamma.grad

    print(dg)
    print((dq - dq2).mean())
    print((dk - dk2).mean())
    print((dv - dv2).mean())
    print((dg - dg2).mean())

    # print((out - out2).abs().mean())
    # print(torch.allclose(out, out2, atol=1e-3, rtol=1e-3))
