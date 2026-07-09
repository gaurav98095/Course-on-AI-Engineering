"""Load test: hit the endpoint with N concurrent users and watch it break.

Closed-loop generator: each of C virtual users fires a request, waits for
the answer, then immediately fires the next — for the whole duration.

Run:   python serve.py                          # terminal 1: the server
       python load_test.py --concurrency 4      # terminal 2: the siege
       python load_test.py --sweep 1 2 4 8 16   # the full story
"""

import argparse
import asyncio
import random
import time

import httpx

QUESTIONS = [
    "Why does an aircraft stall at the critical angle of attack?",
    "How does the attitude indicator work?",
    "What is the difference between induced drag and parasite drag?",
    "How does the pitot-static system feed the airspeed indicator?",
    "What happens to lift when flaps are extended?",
    "What are the errors of the magnetic compass during turns?",
    "How does an altimeter measure altitude?",
    "Why does frost on the wing matter for takeoff?",
]


async def user(client: httpx.AsyncClient, url: str, deadline: float,
               latencies: list, errors: list) -> None:
    """One virtual user: ask, wait, ask again, until time is up."""
    while time.perf_counter() < deadline:
        q = random.choice(QUESTIONS)
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/predict",
                                  json={"question": q, "max_new_tokens": 200})
            r.raise_for_status()
            latencies.append(time.perf_counter() - t0)
        except Exception as e:
            errors.append(type(e).__name__)


def percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(int(p / 100 * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[i]


async def run_level(url: str, concurrency: int, duration: int) -> dict:
    latencies, errors = [], []
    deadline = time.perf_counter() + duration
    async with httpx.AsyncClient(timeout=150) as client:
        await asyncio.gather(*[
            user(client, url, deadline, latencies, errors)
            for _ in range(concurrency)
        ])
    lat = sorted(latencies)
    n = len(lat)
    return {
        "concurrency": concurrency,
        "completed": n,
        "errors": len(errors),
        "throughput_rpm": n / duration * 60,
        "p50": percentile(lat, 50),
        "p95": percentile(lat, 95),
        "p99": percentile(lat, 99),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--sweep", type=int, nargs="+",
                    help="run several levels, e.g. --sweep 1 2 4 8 16")
    ap.add_argument("--duration", type=int, default=180,
                    help="seconds per level (default 180)")
    args = ap.parse_args()

    levels = args.sweep or [args.concurrency]
    print(f"{'conc':>5} {'done':>5} {'err':>4} {'req/min':>8} {'p50 s':>7} {'p95 s':>7} {'p99 s':>7}")
    for c in levels:
        r = await run_level(args.url, c, args.duration)
        print(f"{r['concurrency']:>5} {r['completed']:>5} {r['errors']:>4} "
              f"{r['throughput_rpm']:>8.1f} {r['p50']:>7.1f} {r['p95']:>7.1f} {r['p99']:>7.1f}")


if __name__ == "__main__":
    asyncio.run(main())
