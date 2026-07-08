# Lecture 05 — Prefill, Decode, and the KV Cache

Self-contained: this folder is a **copy-forward** of `04-the-gpu-architecture-and-roofline/` plus a standalone KV cache demo. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 05 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/05-prefill-decode-and-the-kv-cache.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| ⚡ Lightning AI Studio | Everything — `kv_cache.py` needs a GPU and downloads the full 8B model |
| 💻 Your laptop | Browser only, reading the lecture |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Lecture 04

| File | What it is |
| --- | --- |
| `kv_cache.py` | Times generation with the KV cache on vs off, then checks the cache-size formula against measured GPU memory |

`kv_cache.py` downloads the full `Qwen/Qwen3-VL-8B-Instruct` weights (~16 GB) on first run — same model as Lecture 01, so if you've already run that lecture's `rag.py` on this machine, it's cached and this is instant.

(`ingest.py`, `rag.py`, `measure.py`, `serve.py`, `client.py`, `load_test.py`, `roofline.py`, and `data/` are copied forward from Lecture 04 unchanged, so this folder still runs the full system on its own.)

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-1-foundations/05-prefill-decode-and-the-kv-cache
pip install -r requirements.txt
python kv_cache.py
```

The no-cache run is deliberately short (`N_TOKENS = 60`) — without caching, generation cost grows quadratically, so longer runs get slow fast. See Lecture 05, Exercise 1, for pushing this further.

## Troubleshooting

- **CUDA out of memory**: close any other process using the GPU (like a leftover `serve.py`) before running.
- **Speedup ratio looks smaller than the lecture's table**: this is expected at low token counts — the no-cache penalty compounds with sequence length, so a short run understates it. Try Exercise 1.
- Everything from Lectures 01–04's troubleshooting still applies if you also run the RAG system in this folder.

---

Previous: [`../04-the-gpu-architecture-and-roofline/`](../04-the-gpu-architecture-and-roofline/) · Next: [`../06-profiling-where-the-time-actually-goes/`](../06-profiling-where-the-time-actually-goes/) — watching a profiler confirm (or correct) everything this module has claimed so far.
