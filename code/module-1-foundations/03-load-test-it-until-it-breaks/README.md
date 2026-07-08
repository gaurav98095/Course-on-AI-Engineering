# Lecture 03 — Load-Test It Until It Breaks

Self-contained: this folder is a **copy-forward** of `02-deploy-it-on-a-gpu/` plus a load generator and a combined live dashboard. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 03 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/03-load-test-it-until-it-breaks.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| ⚡ Lightning Studio, terminal 1 | `serve.py` — the victim |
| ⚡ Lightning Studio, terminal 2 | `load_test.py` — the swarm (same machine, so we measure the server, not the internet) |
| ⚡ Lightning Studio, terminal 3 | `live_dashboard.py` — watching the API layer and the GPU layer at once |
| 💻 Your laptop | Optional: rerun the swarm over the public URL and compare |

## What's new versus Lecture 02

| File | What it is |
| --- | --- |
| `load_test.py` | Closed-loop async load generator — sweep concurrency, read p50/p95/p99, watch it break |
| `live_dashboard.py` | Polls Lecture 02's `/metrics` route on an interval and prints API-layer counters alongside GPU vitals, side by side, live |

(Everything else — including `gpu_vitals.py`/`plot_vitals.py` from Lecture 01b and `serve.py`'s `/metrics` route from Lecture 02 — is copied forward unchanged so this folder runs on its own.)

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-1-foundations/03-load-test-it-until-it-breaks
pip install -r requirements.txt
python ingest.py                 # build the indexes (skip if you already have corpus/)
```

⚡ Terminal 1:

```bash
python serve.py
```

⚡ Terminal 2 — the sweep:

```bash
python load_test.py --concurrency 1        # sanity check first
python load_test.py --sweep 1 2 4 8 16 32  # watch it break
```

⚡ Terminal 3 — watch both systems at once, live:

```bash
python live_dashboard.py
```

## Troubleshooting

- **Everything errors immediately**: is `serve.py` actually running and past its ~90 s cold start? Check `curl localhost:8000/health` first.
- **Sweep takes forever**: lower `--duration` (default 180 s per level) for a quicker, noisier read.
- **`live_dashboard.py` connection refused**: same cause as above — `serve.py` isn't up yet, or crashed under load. Check terminal 1.
- Everything from Lectures 01/01b/02's troubleshooting applies here too.

---

Previous: [`../02-deploy-it-on-a-gpu/`](../02-deploy-it-on-a-gpu/) · Next: [`../04-the-gpu-architecture-and-roofline/`](../04-the-gpu-architecture-and-roofline/) — explaining the <1% GPU utilization number this lecture just measured.
