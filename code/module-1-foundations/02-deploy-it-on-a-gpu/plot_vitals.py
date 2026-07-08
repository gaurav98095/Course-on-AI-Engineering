"""Plot a vitals.csv from gpu_vitals.py: utilization, memory, and power over time.

Run:  python plot_vitals.py                 # reads vitals.csv, writes vitals.png
      python plot_vitals.py --in run2.csv --out run2.png
"""

import argparse
import csv

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="vitals.csv")
    ap.add_argument("--out", default="vitals.png")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.infile)))
    t = [float(r["t"]) for r in rows]
    util = [float(r["util_gpu_pct"]) for r in rows]
    mem = [float(r["mem_used_mib"]) for r in rows]
    power = [float(r["power_w"]) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, util, color="tab:blue"); axes[0].set_ylabel("GPU util %")
    axes[1].plot(t, mem, color="tab:purple"); axes[1].set_ylabel("Memory (MiB)")
    axes[2].plot(t, power, color="tab:red"); axes[2].set_ylabel("Power (W)")
    axes[2].set_xlabel("seconds")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("GPU vitals over time")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
