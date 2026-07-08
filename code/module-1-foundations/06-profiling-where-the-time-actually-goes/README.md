# Lecture 06 — Profiling: Where the Time Actually Goes

Self-contained: this folder is a **copy-forward** of `05-prefill-decode-and-the-kv-cache/` plus a profiling script. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 06 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/06-profiling-where-the-time-actually-goes.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| ⚡ Lightning AI Studio | Everything — profiling needs a GPU and the full model |
| 💻 Your laptop | Browser only: reading the lecture, and viewing `trace.json` at `https://ui.perfetto.dev` |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Lecture 05

| File | What it is |
| --- | --- |
| `profile_request.py` | Profiles one real request with `torch.profiler`, printing a kernel-level time breakdown and exporting a full trace |

No new Python dependency — `torch.profiler` ships inside `torch`. `nsys`/`ncu` (NVIDIA Nsight Systems/Compute) are separate CLI tools, not Python packages:

- Check availability first: `nsys --version` and `ncu --version`.
- Most CUDA-toolkit-based cloud GPU images ship them already.
- If either is missing, the lecture's `torch.profiler` step (Step 1) already answers the main question on its own — skip straight past the missing tool, nothing else in this folder depends on it.
- `ncu` commonly needs elevated GPU-counter permissions on shared/cloud instances; a `ERR_NVGPUCTRPERM` error is expected in some environments, not a sign anything is broken.

(`ingest.py`, `rag.py`, `measure.py`, `serve.py`, `client.py`, `load_test.py`, `roofline.py`, `kv_cache.py`, and `data/` are copied forward from Lecture 05 unchanged, so this folder still runs the full system on its own.)

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-1-foundations/06-profiling-where-the-time-actually-goes
pip install -r requirements.txt
python profile_request.py
```

Then, optionally, if `nsys`/`ncu` are available on your Studio — see the lecture's Steps 2–3 for the exact commands.

## Troubleshooting

- **`trace.json` won't open**: use `https://ui.perfetto.dev` (works in any browser) rather than `chrome://tracing` if you're not on Chrome.
- **Profiler output looks dominated by warmup**: make sure you're reading the profiled block's output, not a cold first call — `profile_request.py` already warms up before profiling, but a modified script might not.
- Everything from Lectures 01–05's troubleshooting still applies if you also run the RAG system in this folder.

---

Previous: [`../05-prefill-decode-and-the-kv-cache/`](../05-prefill-decode-and-the-kv-cache/) · This is the last lecture in Module 1 — Module 2 (quantization, FlashAttention, serving engines) begins next.
