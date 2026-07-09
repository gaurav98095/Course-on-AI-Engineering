"""Re-run Lecture 03's exact load test -- same questions, same concurrency
sweep, same percentiles -- against vLLM (or SGLang) instead of our naive
one-operator server.

Retrieval is pre-computed once per question, not per request: Lecture 01
and Lecture 03 both already established it costs ~40ms, noise next to
generation, so this isolates what's actually being compared here -- the
serving engine's behavior under concurrency.

Run:   vllm serve Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 \
         --max-model-len 16384 --limit-mm-per-prompt '{"image":2,"video":0}'   # terminal 1
       python load_test_vllm.py --sweep 1 2 4 8 16 32                          # terminal 2
       python load_test_vllm.py --sweep 1 2 4 8 16 32 --port 30000             # against SGLang
"""

import argparse
import asyncio
import random
import time

import httpx

from client_vllm import build_messages
from load_test import QUESTIONS, percentile
from rag import GENERATOR, Retriever


def precompute_messages() -> dict[str, list[dict]]:
    print("pre-computing retrieval for all questions (once, not per request)...")
    retriever = Retriever()
    return {q: build_messages(q, *retriever(q)) for q in QUESTIONS}


async def user(client: httpx.AsyncClient, url: str, messages_by_q: dict, deadline: float,
               latencies: list, errors: list) -> None:
    while time.perf_counter() < deadline:
        q = random.choice(QUESTIONS)
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/chat/completions", json={
                "model": GENERATOR, "messages": messages_by_q[q], "max_tokens": 200,
            })
            r.raise_for_status()
            latencies.append(time.perf_counter() - t0)
        except Exception as e:
            errors.append(type(e).__name__)


async def run_level(url: str, messages_by_q: dict, concurrency: int, duration: int) -> dict:
    latencies, errors = [], []
    deadline = time.perf_counter() + duration
    async with httpx.AsyncClient(timeout=150) as client:
        await asyncio.gather(*[
            user(client, url, messages_by_q, deadline, latencies, errors)
            for _ in range(concurrency)
        ])
    lat = sorted(latencies)
    n = len(lat)
    return {
        "concurrency": concurrency, "completed": n, "errors": len(errors),
        "throughput_rpm": n / duration * 60,
        "p50": percentile(lat, 50), "p95": percentile(lat, 95), "p99": percentile(lat, 99),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000, help="8000 for vLLM, 30000 for SGLang")
    ap.add_argument("--sweep", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--duration", type=int, default=180)
    args = ap.parse_args()
    url = f"http://localhost:{args.port}/v1"

    messages_by_q = precompute_messages()

    print(f"{'conc':>5} {'done':>5} {'err':>4} {'req/min':>8} {'p50 s':>7} {'p95 s':>7} {'p99 s':>7}")
    for c in args.sweep:
        r = await run_level(url, messages_by_q, c, args.duration)
        print(f"{r['concurrency']:>5} {r['completed']:>5} {r['errors']:>4} "
              f"{r['throughput_rpm']:>8.1f} {r['p50']:>7.1f} {r['p95']:>7.1f} {r['p99']:>7.1f}")


if __name__ == "__main__":
    asyncio.run(main())
