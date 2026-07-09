# Lecture 14 — vLLM & SGLang

Self-contained: this folder is a **copy-forward** of `13-continuous-batching/` plus two new files. Clone the repo, `cd` here, and everything runs — this lecture is the one exception in Module 2 that needs a GPU and a real model, since it serves the course model on a real engine instead of simulating one.

Full walkthrough: [Lecture 14 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/14-vllm-and-sglang.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| ⚡ Lightning AI Studio | Everything — `vllm serve`/`sglang.launch_server` needs the GPU; `client_vllm.py` and `load_test_vllm.py` are lightweight HTTP clients that could technically run elsewhere, but keep them in the same Studio for simplicity |
| 💻 Your laptop | Browser only, reading the lecture |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Lecture 13

| File | What it is |
| --- | --- |
| `client_vllm.py` | Lecture 02's `client.py`, adapted: retrieval stays local (unchanged since Lecture 01), generation becomes an OpenAI-compatible API call to vLLM or SGLang instead of an in-process model load |
| `load_test_vllm.py` | Lecture 03's `load_test.py`, re-run against a real engine instead of our naive server — same `QUESTIONS`, same concurrency sweep, same percentile math, imported directly from `load_test.py` |

Neither client script loads the model itself — the GPU work happens inside whichever server process you start (see below). Both vLLM and SGLang speak the same OpenAI-compatible API, so the identical client code works against either; only the port differs (vLLM: 8000, SGLang: 30000).

(Everything from Lectures 01–13 is copied forward unchanged, so this folder still runs the full system, continuous-batching simulator included, on its own.)

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-2-vertical-wins/14-vllm-and-sglang
pip install -r requirements.txt
python ingest.py                    # build corpus/ if you don't have it yet
```

**Install vLLM** (large, hardware-specific — follow [vLLM's own install docs](https://docs.vllm.ai/en/latest/getting_started/installation/) for your exact CUDA version; `pip install vllm>=0.11.0` is the common case, 0.11.0+ specifically needed for Qwen3-VL support):

```bash
pip install "vllm>=0.11.0"
```

⚡ Terminal 1 — start the server:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image":2,"video":0}'
```

⚡ Terminal 2 — once you see `Uvicorn running on http://0.0.0.0:8000`:

```bash
python client_vllm.py "Why does an aircraft stall at the critical angle of attack?"
python load_test_vllm.py --sweep 1 2 4 8 16 32
```

**Optional: the same client code against SGLang instead.** Stop the vLLM server first (one 8B model at a time fits on most single-GPU Studios).

```bash
pip install "sglang[all]"
python -m sglang.launch_server --model Qwen/Qwen3-VL-8B-Instruct --tp 1 --host 0.0.0.0 --port 30000
```

```bash
python client_vllm.py "..." --port 30000
python load_test_vllm.py --sweep 1 2 4 8 16 32 --port 30000
```

## Troubleshooting

- **`vllm serve` fails with an unrecognized model error**: confirm your installed version is `>=0.11.0` (`pip show vllm`) — Qwen3-VL support landed in that release; older versions will not recognize it.
- **OOM on a smaller GPU**: lower `--max-model-len` (context length reserves KV cache up front, Lecture 10's exact lesson) or `--gpu-memory-utilization` (default 0.9 — how much of the card vLLM is allowed to claim).
- **`--limit-mm-per-prompt` errors or is ignored**: this flag's exact JSON shape has changed across vLLM versions — check `vllm serve --help` on your installed version if the example above doesn't parse.
- **`load_test_vllm.py` errors immediately with a connection refused**: the server takes time to load (same ~90s-class cold start as Lecture 02's `serve.py`) — wait for the `Uvicorn running` line before starting the client.
- **Numbers don't match this lecture's**: expected and explicitly flagged in the lecture itself — every number in Measure It is either vLLM's own independently published benchmark or explicitly marked as something to run yourself; nothing here was executed on real hardware before publishing.
- Everything from Lectures 01–13's troubleshooting still applies if you also run those systems in this folder.

---

Previous: [`../13-continuous-batching/`](../13-continuous-batching/) · This is the last lecture published in Module 2 so far — more lectures land here as the course continues.
