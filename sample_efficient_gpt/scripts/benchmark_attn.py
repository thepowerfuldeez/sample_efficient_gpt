import torch
from triton.testing import runtime, _summarize_statistics
from prettytable import PrettyTable
import numpy as np

torch.cuda.set_device("cuda:0")
torch.set_float32_matmul_precision("high")

from sample_efficient_gpt.transformer.triton_flash_attn import TritonFlashAttnFunc
from sample_efficient_gpt.transformer.triton_flash_attn_qknorm import TritonFlashAttnQKNormFunc


def do_bench(fn, args, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".
    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()

    fn(*args)
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        runtime.driver.active.clear_cache(cache)
        fn(*args)
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn(*args)
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        fn(*args)
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)


def attn_qknorm(Q, K, V, gain, q_tile, k_tile, backward_pass=False):
    seq_len = Q.shape[1]
    if backward_pass:
        for t in (Q, K, V):
            t.grad = None
            t.requires_grad_(True)
        out = TritonFlashAttnQKNormFunc.apply(Q, K, V, gain, True, q_tile, k_tile)
        out.sum().backward()
    else:
        out = TritonFlashAttnQKNormFunc.apply(Q, K, V, gain, True, q_tile, k_tile)


def flash_attn(Q, K, V, q_tile, k_tile, backward_pass=False):
    if backward_pass:
        for t in (Q, K, V):
            t.grad = None
            t.requires_grad_(True)
        out = TritonFlashAttnFunc.apply(Q, K, V, True, q_tile, k_tile)
        loss = out.sum()
        loss.backward()
    else:
        out = TritonFlashAttnFunc.apply(Q, K, V, True, q_tile, k_tile)


def _make_attn_inputs(batch_size, n_queries, n_keys, D, device=None):
    torch.random.manual_seed(0)
    q = torch.randn(batch_size, n_queries, D, dtype=torch.bfloat16, device=device, requires_grad=True)
    k = torch.randn(batch_size, n_keys, D, dtype=torch.bfloat16, device=device, requires_grad=True)
    v = torch.randn(batch_size, n_keys, D, dtype=torch.bfloat16, device=device, requires_grad=True)
    gain = torch.randn(1, dtype=torch.bfloat16, device=device, requires_grad=True)

    return q, k, v, gain


if __name__ == "__main__":
    BACKWARD_PASS = True

    hd = 64
    nh = 32
    
    # for q_tile, k_tile in [(32, 32), (64, 64), (128, 64), (128, 128)]:
    for q_tile, k_tile in [(64, 64)]:
        table = PrettyTable(title=f"Q tile: {q_tile}, K tile: {k_tile}")
        table.field_names = ["Sequence", "QKNorm Flash attn", "Flash attn", "TFLOP/s", "Speedup"]
        speedups = []
        for seq in [512, 1024, 2048, 4096, 8192]: 
            bs = (16384 // seq)
            Q, K, V, gain = _make_attn_inputs(bs * nh, seq, seq, hd, "cuda")

            flop = 2 * seq * seq * hd * nh * bs
            if BACKWARD_PASS:
                flop = flop + 2.5 * flop

            flash_attn_time = do_bench(flash_attn, (Q, K, V, q_tile, k_tile, BACKWARD_PASS), warmup=50)
            qknorm_attn_time = do_bench(attn_qknorm, (Q, K, V, gain, q_tile, k_tile, BACKWARD_PASS), warmup=50)

            tflops = flop / (flash_attn_time * 1e-3) / 1e12
            speedup = 1 / (flash_attn_time / qknorm_attn_time)
            speedups.append(speedup)
            table.add_row([seq, qknorm_attn_time, flash_attn_time, tflops, speedup])
        print(table)
        print(f"Avg speedup: {np.mean(speedups):.4f}")

    # for q_tile, k_tile in [(16, 16), (32, 16), (32, 32)]:
    #     table = PrettyTable(title=f"Q tile: {q_tile}, K tile: {k_tile}")
    #     table.field_names = ["Sequence", "Head dim", "Attn", "Flash attn", "Speedup"]
    #     speedups = []
    #     for seq in [128, 256, 512, 1024, 2048, 4096, 8192, 16384] + ([32768] if not BACKWARD_PASS else []): 
    #         for hd in [32, 64, 128]:
    #             Q, K, V = _make_attn_inputs(1, seq, seq, hd, "cuda")

    #             attn_time = do_bench(attn, (Q, K, V, BACKWARD_PASS), warmup=50)
    #             flash_attn_time = do_bench(flash_attn, (Q, K, V, q_tile, k_tile, BACKWARD_PASS), warmup=50)
    #             speedup = 1 / (flash_attn_time / attn_time)
    #             speedups.append(speedup)
    #             table.add_row([seq, hd, attn_time, flash_attn_time, speedup])
    #     print(table)
    #     print(f"Avg speedup: {np.mean(speedups):.4f}")