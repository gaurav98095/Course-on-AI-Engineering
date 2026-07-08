# Lecture 01b — GPU Vitals: Watching What You Built

Self-contained: this folder is a **copy-forward** of `01-build-a-multimodal-rag/` plus continuous GPU monitoring tools. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 01b on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/01b-gpu-vitals.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| ⚡ Lightning AI Studio | Everything — two terminals: one running `rag.py`, one watching vitals |
| 💻 Your laptop | Browser only, reading the lecture |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Lecture 01

| File | What it is |
| --- | --- |
| `gpu_vitals.py` | Continuous GPU monitor — polls utilization, memory, power, temperature, clock via `pynvml`, logs to CSV |
| `plot_vitals.py` | Plots a `vitals.csv` into a 3-panel chart (utilization / memory / power) |

(`ingest.py`, `rag.py`, `measure.py`, and `data/` are copied forward from Lecture 01 unchanged, so this folder still runs the full system on its own.)

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-1-foundations/01b-gpu-vitals
pip install -r requirements.txt
python ingest.py                 # build the indexes (skip if you already have corpus/)
```

⚡ Terminal 1 — the zero-install path (no Python needed):

```bash
nvidia-smi dmon
```

⚡ Terminal 2 — fire a request and watch terminal 1 spike:

```bash
python rag.py "Why does an aircraft stall at the critical angle of attack?"
```

Then the Python-native path, so you can log and plot:

⚡ Terminal 1:

```bash
python gpu_vitals.py --seconds 30
```

⚡ Terminal 2, while that's running:

```bash
python rag.py "How does the attitude indicator work?"
```

Then plot it:

```bash
python plot_vitals.py
```

## Troubleshooting

- **`pynvml` import error**: it's a thin wrapper over the NVIDIA driver's management library — make sure you're on a machine with an NVIDIA GPU and driver, not a CPU-only environment.
- **All zeros in the log**: confirm you're polling the right GPU index (`nvmlDeviceGetHandleByIndex(0)` assumes GPU 0 — fine on a single-GPU Studio).
- Everything from Lecture 01's troubleshooting applies here too.

---

Previous: [`../01-build-a-multimodal-rag/`](../01-build-a-multimodal-rag/) · Next: [`../02-deploy-it-on-a-gpu/`](../02-deploy-it-on-a-gpu/) — these same vitals, exposed over HTTP.
