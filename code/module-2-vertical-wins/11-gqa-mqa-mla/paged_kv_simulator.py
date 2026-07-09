"""Three real numbers from one idea: stop pre-reserving KV cache memory
for the worst case, and start allocating it in small blocks, on demand --
the way an OS pages virtual memory. No GPU needed: this is a bookkeeping
simulation, not a benchmark, and every number it prints is real arithmetic
you can check by hand.

Three demonstrations:
  1. Internal fragmentation -- naive (reserve max_model_len per request)
     vs paged (reserve in BLOCK_SIZE-token chunks, on demand).
  2. Prefix sharing -- our own RAG system's system prompt (and often its
     retrieved excerpts) is identical across many requests; paging lets
     those tokens' blocks be stored once and referenced by every request.
  3. The same lever stacked twice -- paging, plus Lecture 07's int8
     quantization applied to the cache tensors themselves.

Run:  python paged_kv_simulator.py
"""

import math
import random

from transformers import AutoConfig

from kv_cache import cache_size_bytes

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MAX_MODEL_LEN = 8192   # what a naive allocator must reserve per request, worst case
BLOCK_SIZE = 16         # vLLM's real default block size
N_REQUESTS = 1000
SEED = 42


def sample_length(rng: random.Random) -> int:
    """A realistic traffic mix: mostly short turns, some long-context RAG, rare near-max."""
    r = rng.random()
    if r < 0.80:
        return rng.randint(50, 800)
    elif r < 0.95:
        return rng.randint(800, 3000)
    return rng.randint(3000, 8000)


def fragmentation_demo(cfg) -> tuple[int, int]:
    rng = random.Random(SEED)
    lengths = [sample_length(rng) for _ in range(N_REQUESTS)]
    used_tokens = sum(lengths)

    naive_reserved = N_REQUESTS * MAX_MODEL_LEN
    naive_waste_pct = (naive_reserved - used_tokens) / naive_reserved * 100

    paged_reserved = sum(math.ceil(l / BLOCK_SIZE) * BLOCK_SIZE for l in lengths)
    paged_waste_pct = (paged_reserved - used_tokens) / paged_reserved * 100

    print(f"--- 1. Internal fragmentation ({N_REQUESTS} requests, mean length "
          f"{used_tokens / N_REQUESTS:.0f} tokens, max_model_len={MAX_MODEL_LEN}) ---")
    print(f"naive:  reserves {naive_reserved:>10,} tokens-worth for {used_tokens:,} used "
          f"-> {naive_waste_pct:.1f}% wasted")
    print(f"paged:  reserves {paged_reserved:>10,} tokens-worth for {used_tokens:,} used "
          f"-> {paged_waste_pct:.1f}% wasted  (block_size={BLOCK_SIZE})")
    print(f"reservation shrinks {naive_reserved / paged_reserved:.1f}x for the identical batch\n")

    naive_gib = naive_reserved * cache_size_bytes(cfg, tokens=1) / 2**30
    paged_gib = paged_reserved * cache_size_bytes(cfg, tokens=1) / 2**30
    print(f"in real VRAM (course model, bf16 cache): naive={naive_gib:.1f} GiB, "
          f"paged={paged_gib:.1f} GiB\n")
    return naive_reserved, paged_reserved


def prefix_sharing_demo() -> None:
    n_requests, shared_prefix, unique_suffix = 50, 500, 50
    without_sharing = n_requests * (shared_prefix + unique_suffix)
    with_sharing = shared_prefix + n_requests * unique_suffix   # shared blocks stored once
    savings_pct = (without_sharing - with_sharing) / without_sharing * 100

    print(f"--- 2. Prefix sharing ({n_requests} requests, {shared_prefix}-token shared "
          f"prefix, {unique_suffix}-token unique suffix each) ---")
    print("Lecture 01's SYSTEM message -- and often its retrieved excerpts, for a popular "
          "question -- is identical across many requests. Right now every request pays for "
          "it again.")
    print(f"without sharing: {without_sharing:,} tokens-worth stored")
    print(f"with sharing:    {with_sharing:,} tokens-worth stored  -> {savings_pct:.1f}% saved\n")


def quantized_cache_demo(cfg, paged_reserved: int, naive_reserved: int) -> None:
    bf16_gib = paged_reserved * cache_size_bytes(cfg, tokens=1, dtype_bytes=2) / 2**30
    int8_gib = paged_reserved * cache_size_bytes(cfg, tokens=1, dtype_bytes=1) / 2**30
    naive_bf16_gib = naive_reserved * cache_size_bytes(cfg, tokens=1, dtype_bytes=2) / 2**30

    print("--- 3. The same lever, stacked twice: paging + int8 cache ---")
    print(f"paged, bf16 cache (s=2, Lecture 05's format): {bf16_gib:.1f} GiB")
    print(f"paged, int8 cache (s=1, Lecture 07's lever, applied to a different tensor): "
          f"{int8_gib:.1f} GiB")
    print(f"vs. naive + bf16 baseline ({naive_bf16_gib:.1f} GiB): "
          f"{naive_bf16_gib / int8_gib:.1f}x smaller, combined\n")


def main() -> None:
    cfg = AutoConfig.from_pretrained(MODEL)   # downloads config.json only -- seconds, not minutes
    naive_reserved, paged_reserved = fragmentation_demo(cfg)
    prefix_sharing_demo()
    quantized_cache_demo(cfg, paged_reserved, naive_reserved)


if __name__ == "__main__":
    main()
