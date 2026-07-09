# Lecture 13 — Continuous Batching

Self-contained: this folder is a **copy-forward** of `12-rope-alibi-yarn/` plus one new file. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 13 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/13-continuous-batching.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| 💻 Your laptop | `continuous_batching_simulator.py` needs no GPU and no extra package — a pure standard-library scheduling simulation |
| ⚡ Lightning AI Studio | Everything else from earlier lectures still needs a GPU |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Lecture 12

| File | What it is |
| --- | --- |
| `continuous_batching_simulator.py` | Simulates 200 requests with realistic, varied response lengths under two scheduling policies — static batching (wait for a whole batch's slowest member) and continuous batching (evict and backfill every step) — and measures real wall time and GPU-slot utilization for both |

This script needs **no GPU and no third-party package at all** — pure Python standard library (`random`), a scheduling simulation, not a benchmark. Every number it prints is real, reproducible output.

(Everything from Lectures 01–12 is copied forward unchanged, so this folder still runs the full system, RoPE/ALiBi/YaRN demonstrations included, on its own.)

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-2-vertical-wins/13-continuous-batching
pip install -r requirements.txt
```

```bash
python continuous_batching_simulator.py    # no GPU, no download, no package needed at all
```

## Troubleshooting

- **Your own numbers differ from the lecture's**: expected — they depend on `N_REQUESTS`, `BATCH_SIZE`, `SEED`, and the length distribution `sample_length` draws from, all editable at the top of the script. The *shape* of the result (continuous batching wins, by a lot, especially with varied response lengths) is the point, not the exact ratio.
- **Want to see this for real, not simulated?** That's exactly what vLLM's and TGI's real schedulers implement — Lecture 14 serves the course model on vLLM directly and load-tests it the way Lecture 03 did the naive server.
- Everything from Lectures 01–12's troubleshooting still applies if you also run those systems in this folder.

---

Previous: [`../12-rope-alibi-yarn/`](../12-rope-alibi-yarn/) · Next: [`../14-vllm-and-sglang/`](../14-vllm-and-sglang/)
