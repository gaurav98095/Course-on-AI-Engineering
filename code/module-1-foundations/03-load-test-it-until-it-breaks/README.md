# Lecture 03 — Load-Test It Until It Breaks

Self-contained: this folder is a **copy-forward** of `02-deploy-it-on-a-gpu/` plus a load generator. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 03 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/03-load-test-it-until-it-breaks.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| ⚡ Lightning Studio, terminal 1 | `serve.py` — the victim |
| ⚡ Lightning Studio, terminal 2 | `load_test.py` — the swarm (same machine, so we measure the server, not the internet) |
| 💻 Your laptop | Optional: rerun the swarm over the public URL and compare |

## What's new versus Lecture 02

| File | What it is |
| --- | --- |
| `load_test.py` | Closed-loop async load generator — sweep concurrency, read p50/p95/p99, watch it break |

(Everything else is copied forward unchanged so this folder runs on its own.)

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

⚡ Terminal 3 — watch the GPU lie about being busy:

```bash
watch -n 1 nvidia-smi
```

## Troubleshooting

- **Everything errors immediately**: is `serve.py` actually running and past its ~90 s cold start? Check `curl localhost:8000/health` first.
- **Sweep takes forever**: lower `--duration` (default 180 s per level) for a quicker, noisier read.
- Everything from Lectures 01–02's troubleshooting applies here too.

---

Previous: [`../02-deploy-it-on-a-gpu/`](../02-deploy-it-on-a-gpu/) · This is the last lecture published in Module 1 so far — more lectures land here as the course continues.
