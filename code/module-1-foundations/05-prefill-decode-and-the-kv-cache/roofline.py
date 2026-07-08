"""Roofline: measure this GPU's real ceilings, then locate our own workloads on them.

Empirically measures achievable compute (TFLOP/s) and memory bandwidth
(GB/s) on THIS GPU, then times a decode-shaped op (1 token) and a
prefill-shaped op (many tokens at once) and reports which side of the
ridge point each one falls on.

Run:  python roofline.py
"""

import time

import torch
from transformers import AutoConfig

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_HIDDEN = 4096   # used only if the real config can't be fetched
DTYPE = torch.float16
N_LAYERS_SIM = 12       # distinct weight matrices, rotated every step so the
                        # decode timing can't hide in L2 cache like a single
                        # reused matrix would


def get_hidden_size() -> int:
    """Read the real hidden size from the course model's config — don't guess it."""
    try:
        cfg = AutoConfig.from_pretrained(MODEL)
        text_cfg = getattr(cfg, "text_config", cfg)
        d = getattr(text_cfg, "hidden_size", None)
        if d:
            print(f"hidden_size read from {MODEL}'s real config: {d}")
            return d
    except Exception as e:
        print(f"couldn't fetch model config ({e}); using default {DEFAULT_HIDDEN}")
    return DEFAULT_HIDDEN


def timed(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def measure_compute_ceiling(dim: int = 8192) -> float:
    """A big square matmul: so much reuse per byte that this is pure compute."""
    a = torch.randn(dim, dim, dtype=DTYPE, device="cuda")
    b = torch.randn(dim, dim, dtype=DTYPE, device="cuda")
    t = timed(lambda: a @ b, warmup=5, iters=20)
    flops = 2 * dim**3
    return flops / t / 1e12   # TFLOP/s


def measure_bandwidth_ceiling(n_elem: int = 256 * 1024 * 1024) -> float:
    """A large elementwise add: this workload is pure memory traffic."""
    a = torch.randn(n_elem, dtype=DTYPE, device="cuda")
    b = torch.randn(n_elem, dtype=DTYPE, device="cuda")
    t = timed(lambda: a.add_(b), warmup=5, iters=20)
    bytes_moved = 3 * n_elem * a.element_size()   # read a, read b, write a
    return bytes_moved / t / 1e9   # GB/s


def measure_workload(tokens: int, d: int) -> dict:
    """One linear layer, `tokens` rows at once: decode is tokens=1, prefill is tokens>>1.

    Rotates through N_LAYERS_SIM distinct weight matrices so the timing reflects
    real HBM traffic, not an L2-cache-resident toy matrix.
    """
    weights = [torch.randn(d, d, dtype=DTYPE, device="cuda") for _ in range(N_LAYERS_SIM)]
    x = torch.randn(tokens, d, dtype=DTYPE, device="cuda")
    i = 0

    def step():
        nonlocal i
        _ = x @ weights[i % N_LAYERS_SIM]
        i += 1

    t = timed(step, warmup=N_LAYERS_SIM, iters=N_LAYERS_SIM * 4)

    flops = 2 * tokens * d * d                       # one GEMM: rows x (d x d)
    bytes_moved = 2 * d * d + 2 * 2 * tokens * d      # weight (dominant) + read x + write y
    ai = flops / bytes_moved
    return {
        "tokens": tokens, "t_ms": t * 1000, "ai": ai,
        "tflops": flops / t / 1e12, "gbs": bytes_moved / t / 1e9,
    }


def main() -> None:
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    print("measuring compute ceiling (large square matmul, fp16 tensor core)...")
    peak_tflops = measure_compute_ceiling()
    print(f"  achieved: {peak_tflops:.1f} TFLOP/s\n")

    print("measuring bandwidth ceiling (large elementwise add)...")
    peak_gbs = measure_bandwidth_ceiling()
    print(f"  achieved: {peak_gbs:.0f} GB/s\n")

    ridge = peak_tflops * 1e12 / (peak_gbs * 1e9)
    print(f"ridge point on THIS run: {ridge:.0f} FLOPs/byte "
          f"(workloads below this are memory-bound, above it are compute-bound)\n")

    d = get_hidden_size()
    print(f"\n{'tokens':>7} {'AI (F/B)':>10} {'time ms':>9} {'TFLOP/s':>9} {'GB/s':>7} {'regime':>13}")
    for tokens in [1, 2, 8, 32, 128, 512, 2048]:
        r = measure_workload(tokens, d)
        regime = "memory-bound" if r["ai"] < ridge else "compute-bound"
        print(f"{r['tokens']:>7} {r['ai']:>10.1f} {r['t_ms']:>9.3f} "
              f"{r['tflops']:>9.2f} {r['gbs']:>7.0f} {regime:>13}")


if __name__ == "__main__":
    main()
