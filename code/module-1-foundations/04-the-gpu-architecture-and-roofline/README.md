# Lecture 04 — The GPU: Architecture, HBM, and the Roofline Model

Self-contained: this folder is a **copy-forward** of `03-load-test-it-until-it-breaks/` plus a standalone GPU benchmark. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 04 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/04-the-gpu-architecture-and-roofline.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| ⚡ Lightning AI Studio | Everything — `roofline.py` needs a GPU |
| 💻 Your laptop | Browser only, reading the lecture |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Lecture 03

| File | What it is |
| --- | --- |
| `roofline.py` | Measures your GPU's real compute and bandwidth ceilings, then sweeps decode-shaped to prefill-shaped workloads across them |

`roofline.py` does **not** need `corpus/` built or the server running — it's a standalone GPU microbenchmark. It does fetch `Qwen/Qwen3-VL-8B-Instruct`'s config (a few KB, not the weights) to read the model's real hidden size.

(`ingest.py`, `rag.py`, `measure.py`, `serve.py`, `client.py`, `load_test.py`, and `data/` are copied forward from Lecture 03 unchanged, so this folder still runs the full system on its own.)

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-1-foundations/04-the-gpu-architecture-and-roofline
pip install -r requirements.txt
python roofline.py
```

Takes under a minute — no model weights downloaded, no indexing required.

## Troubleshooting

- **CUDA not available**: `roofline.py` requires a GPU; confirm with `nvidia-smi` first.
- **Numbers look much lower than the lecture's table**: check nothing else is using the GPU (`nvidia-smi`) — a concurrent process (like a leftover `serve.py`) will steal bandwidth and compute.
- Everything from Lectures 01–03's troubleshooting still applies if you also run the RAG system in this folder.

---

Previous: [`../03-load-test-it-until-it-breaks/`](../03-load-test-it-until-it-breaks/) · Next: [`../05-prefill-decode-and-the-kv-cache/`](../05-prefill-decode-and-the-kv-cache/) — placing prefill and decode as two literal points on this lecture's roofline.
