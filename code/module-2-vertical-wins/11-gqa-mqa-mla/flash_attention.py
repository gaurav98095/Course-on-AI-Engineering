"""Naive attention vs PyTorch's fused SDPA -- same math, radically
different memory traffic.

Naive attention materializes the full N x N score matrix in HBM before
ever reducing it to an output. SDPA (scaled_dot_product_attention) fuses
the whole computation into one kernel -- on supported hardware/dtypes this
dispatches to a FlashAttention-style backend that never writes that N x N
matrix to HBM at all. Same output, same FLOP count in the ballpark, very
different bytes moved -- which is exactly the roofline-model lever from
Lecture 04.

Run:  python flash_attention.py
"""

import time

import torch

B, H, D = 1, 32, 128   # course model's query-head count / head_dim (config.json) -- Lecture 05's
                       # GQA head-sharing is a separate lever (Lecture 11); this benchmark uses
                       # plain equal-head Q/K/V shapes, the case flash attention was built for
SEQ_LENS = [512, 1024, 2048, 4096, 8192]


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / (d ** 0.5)   # (B, H, N, N) -- this is the HBM traffic flash attention avoids
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)   # PyTorch picks the fused kernel


def timed(fn, *args, warmup: int = 5, iters: int = 20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    return elapsed, peak_mib


def main() -> None:
    torch.manual_seed(0)
    print(f"{'seq_len':>8} {'naive ms':>10} {'naive MiB':>10} {'flash ms':>10} {'flash MiB':>10} "
          f"{'speedup':>9} {'mem ratio':>10}")

    for seq_len in SEQ_LENS:
        q = torch.randn(B, H, seq_len, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, H, seq_len, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, H, seq_len, D, dtype=torch.float16, device="cuda")

        try:
            t_naive, m_naive = timed(naive_attention, q, k, v)
            naive_str = f"{t_naive * 1000:>10.2f} {m_naive:>10.1f}"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            t_naive, m_naive = None, None
            naive_str = f"{'OOM':>10} {'OOM':>10}"

        t_flash, m_flash = timed(flash_attention, q, k, v)
        flash_str = f"{t_flash * 1000:>10.2f} {m_flash:>10.1f}"

        if t_naive is not None:
            ratio_str = f"{t_naive / t_flash:>8.1f}x {m_naive / m_flash:>9.1f}x"
        else:
            ratio_str = f"{'--':>9} {'--':>10}"

        print(f"{seq_len:>8} {naive_str} {flash_str} {ratio_str}")


if __name__ == "__main__":
    main()
