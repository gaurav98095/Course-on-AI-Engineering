"""Watch the API layer and the GPU layer at the same time, live, during a load test.

Polls serve.py's /metrics endpoint (added in Lecture 02 — it already combines
API-layer counters and GPU vitals in one response) on an interval, so you can
watch both sides of the "two systems" split from the same terminal while
load_test.py hammers the server in another one.

Run:   python serve.py                                  # terminal 1
       python load_test.py --sweep 1 2 4 8 16 32         # terminal 2
       python live_dashboard.py                          # terminal 3, while both run
"""

import argparse
import csv
import time

import requests

FIELDS = ["t", "requests_total", "errors_total", "latency_p50_s",
          "gpu_util_pct", "gpu_mem_used_mib"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", default="dashboard.csv")
    args = ap.parse_args()

    t0 = time.time()
    print(f"polling {args.url}/metrics every {args.interval}s -> {args.out} (Ctrl-C to stop)\n")
    print(f"{'t':>7}  {'API layer':<32}  ||  {'GPU layer'}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        try:
            while True:
                m = requests.get(f"{args.url}/metrics", timeout=5).json()
                row = {"t": round(time.time() - t0, 1),
                       **{k: m[k] for k in FIELDS if k != "t"}}
                writer.writerow(row)
                f.flush()
                print(f"t={row['t']:>6.1f}s  requests={row['requests_total']:>4}  "
                      f"errors={row['errors_total']:>3}  p50={row['latency_p50_s']:>6.1f}s   ||   "
                      f"gpu_util={row['gpu_util_pct']:>3}%  gpu_mem={row['gpu_mem_used_mib']:>7.0f}MiB")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass

    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
