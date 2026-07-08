"""GPU vitals: a continuous monitor, not a single glance.

Two ways to watch, cheapest first:
  1. `nvidia-smi dmon` — built into every NVIDIA driver, zero install.
  2. This script — pynvml polling at a fixed interval, logged to CSV, so you
     can plot it later and correlate vitals against what was actually running.

Run:  python gpu_vitals.py                    # logs to vitals.csv until Ctrl-C
      python gpu_vitals.py --seconds 30        # logs for a fixed window
      python gpu_vitals.py --interval 0.1      # sample 10x/sec instead of 1x/sec
"""

import argparse
import csv
import time

import pynvml

FIELDS = ["t", "util_gpu_pct", "util_mem_pct", "mem_used_mib", "mem_total_mib",
          "power_w", "temp_c", "sm_clock_mhz"]


def sample(handle) -> dict:
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0     # mW -> W
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
    return {
        "util_gpu_pct": util.gpu,
        "util_mem_pct": util.memory,
        "mem_used_mib": mem.used / 2**20,
        "mem_total_mib": mem.total / 2**20,
        "power_w": power,
        "temp_c": temp,
        "sm_clock_mhz": clock,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long (default: run until Ctrl-C)")
    ap.add_argument("--out", default="vitals.csv")
    args = ap.parse_args()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode()
    print(f"watching: {name}  (sampling every {args.interval}s -> {args.out}, Ctrl-C to stop)")

    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        try:
            while args.seconds is None or (time.time() - t0) < args.seconds:
                row = {"t": round(time.time() - t0, 2), **sample(handle)}
                writer.writerow(row)
                f.flush()
                print(f"t={row['t']:>6.1f}s  util={row['util_gpu_pct']:>3}%  "
                      f"mem={row['mem_used_mib']:>7.0f}/{row['mem_total_mib']:.0f} MiB  "
                      f"power={row['power_w']:>5.1f}W  temp={row['temp_c']}C  "
                      f"sm_clock={row['sm_clock_mhz']}MHz")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass

    pynvml.nvmlShutdown()
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
