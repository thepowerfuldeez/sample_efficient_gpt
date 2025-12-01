import torch
import triton
import triton.language as tl
import time


def print_device_info():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")


def benchmark(fn, *args, warmup=10, rep=100):
    # Warmup (GPU lazy init, cache)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(rep):
        fn(*args)
    torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000 / rep
    return avg_ms


def get_autotune_config():
    return [
        # Ultra extreme split-K for small problems
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16, "SPLIT_K": 32}, num_stages=1, num_warps=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16, "SPLIT_K": 32}, num_stages=1, num_warps=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 16, "SPLIT_K": 32}, num_stages=1, num_warps=1),
        # Extreme split-K for medium problems
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 16, "SPLIT_K": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 16, "SPLIT_K": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32, "SPLIT_K": 16}, num_stages=2, num_warps=1),
        # Heavy split-K for 8k single
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 16, "SPLIT_K": 8}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 32, "SPLIT_K": 8}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 16, "SPLIT_K": 8}, num_stages=3, num_warps=4),
        # Optimized for multi-group
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 32, "SPLIT_K": 4}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 16, "BLOCK_K": 32, "SPLIT_K": 2}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 64, "SPLIT_K": 4}, num_stages=3, num_warps=2),
        # Large blocks for biggest problems
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 16, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=3, num_warps=4),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def grouped_gemm_kernel_optimized(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    GROUPS,
    stride_ag,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cg,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = SPLIT_K

    num_pid_in_group = num_pid_m * num_pid_n * num_pid_k
    group_id = pid // num_pid_in_group

    if group_id >= GROUPS:
        return

    pid_in_group = pid % num_pid_in_group
    pid_k = pid_in_group % num_pid_k
    pid_mn = pid_in_group // num_pid_k

    # Swizzle for cache optimization
    GROUP_M = 8
    pid_m = (pid_mn % GROUP_M) * (num_pid_m // GROUP_M) + (pid_mn // GROUP_M // num_pid_n)
    pid_n = (pid_mn // GROUP_M) % num_pid_n

    if SPLIT_K > 1:
        k_start = pid_k * tl.cdiv(K, SPLIT_K)
        k_end = tl.minimum(k_start + tl.cdiv(K, SPLIT_K), K)
    else:
        k_start = 0
        k_end = K

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = k_start + tl.arange(0, BLOCK_K)

    a_base = A_ptr + group_id * stride_ag + offs_m[:, None] * stride_am
    b_base = B_ptr + group_id * stride_bg + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(k_start, k_end, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_base + offs_k[None, :] * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(b_base + offs_k[:, None] * stride_bk, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)
        offs_k += BLOCK_K

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_base = C_ptr + group_id * stride_cg
    c_ptrs = c_base + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, c_mask)


def grouped_gemm_naive(A, B):
    """Naive PyTorch implementation using a for loop"""
    G, M, K = A.shape
    _, K_, N = B.shape
    assert K == K_

    C = torch.empty((G, M, N), device=A.device, dtype=A.dtype)
    for g in range(G):
        C[g] = torch.matmul(A[g], B[g])
    return C


def grouped_gemm(A, B):
    G, M, K = A.shape
    _, K_, N = B.shape
    assert K == K_

    # Custom configuration for each size
    if M == 512:
        # Small - use extreme split-K
        C = torch.zeros((G, M, N), device=A.device, dtype=torch.float32)
        grid = lambda META: (G * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]) * META["SPLIT_K"],)
        grouped_gemm_kernel_optimized[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            G,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
        )
        C = C.to(A.dtype)
    elif M == 2048:
        # Medium - use heavy split-K
        C = torch.zeros((G, M, N), device=A.device, dtype=torch.float32)
        grid = lambda META: (G * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]) * META["SPLIT_K"],)
        grouped_gemm_kernel_optimized[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            G,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
        )
        C = C.to(A.dtype)
    elif M == 8192 and G == 1:
        # Single 8k - maximum split-K
        C = torch.zeros((G, M, N), device=A.device, dtype=torch.float32)
        grid = lambda META: (G * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]) * META["SPLIT_K"],)
        grouped_gemm_kernel_optimized[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            G,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
        )
        C = C.to(A.dtype)
    elif M == 8192 and G == 4:
        # Multi-group 8k - moderate split-K
        C = torch.zeros((G, M, N), device=A.device, dtype=torch.float32)
        grid = lambda META: (G * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]) * META["SPLIT_K"],)
        grouped_gemm_kernel_optimized[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            G,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
        )
        C = C.to(A.dtype)
    else:
        # Large - standard approach
        C = torch.empty((G, M, N), device=A.device, dtype=A.dtype)
        grid = lambda META: (G * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]) * META["SPLIT_K"],)
        grouped_gemm_kernel_optimized[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            G,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
        )

    return C


def run_benchmarks():
    print_device_info()

    configs = [
        # G, M, K, N  ->  G=groups (like microbatches/layers), M=tokens, K=hidden, N=num_experts
        (1, 512, 896, 16),  # small batch
        (1, 2048, 896, 16),  # medium batch
        (1, 8192, 896, 16),  # 8k tokens (typical 8×1024)
        (4, 8192, 896, 16),  # multiple layers/groups
        (8, 16384, 896, 16),  # large batch
    ]

    for G, M, K, N in configs:
        A = torch.randn(G, M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(G, K, N, device="cuda", dtype=torch.bfloat16)

        # Warm-up JIT and cache
        grouped_gemm(A, B)
        torch.bmm(A, B)
        grouped_gemm_naive(A, B)

        triton_ms = benchmark(grouped_gemm, A, B, warmup=50, rep=200)
        torch_ms = benchmark(torch.bmm, A, B, warmup=50, rep=200)
        naive_ms = benchmark(grouped_gemm_naive, A, B, warmup=50, rep=200)

        gflops = 2 * G * M * N * K / 1e9
        print(f"G={G}, shape=({M},{K},{N})")
        print(f" Triton: {triton_ms:.3f} ms, {gflops / (triton_ms / 1e3):.2f} GFLOP/s")
        print(f" Torch : {torch_ms:.3f} ms, {gflops / (torch_ms / 1e3):.2f} GFLOP/s")
        print(f" Naive : {naive_ms:.3f} ms, {gflops / (naive_ms / 1e3):.2f} GFLOP/s")
        print(f" Triton vs Torch: {torch_ms / triton_ms:.2f}x")
        print(f" Triton vs Naive: {naive_ms / triton_ms:.2f}x")
        print(f" Torch vs Naive: {naive_ms / torch_ms:.2f}x\n")


if __name__ == "__main__":
    run_benchmarks()
