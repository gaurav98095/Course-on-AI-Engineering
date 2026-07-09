# Lecture 10 — PagedAttention & the KV Cache Pool

Self-contained: this folder is a **copy-forward** of `09-flashattention/` plus one new file. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 10 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/10-pagedattention-kv-cache-pool.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| 💻 Your laptop | `paged_kv_simulator.py` needs no GPU at all — a bookkeeping simulation plus one `config.json` download runs anywhere |
| ⚡ Lightning AI Studio | Everything else from earlier lectures still needs a GPU |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Lecture 09

| File | What it is |
| --- | --- |
| `paged_kv_simulator.py` | Three real, runnable demonstrations of paged KV cache allocation: (1) internal fragmentation, naive pre-allocation vs paging in fixed blocks, (2) prefix sharing across requests with a common system prompt, (3) int8 cache quantization stacked on top of paging |

This script needs **no GPU** — it's pure arithmetic plus one `AutoConfig.from_pretrained` call to read the course model's real layer/head config, the same lightweight download `roofline.py` (Lecture 04) and `kv_cache.py` (Lecture 05) already use. Every number it prints is real, reproducible output — run it on your laptop.

(Everything from Lectures 01–09 is copied forward unchanged, so this folder still runs the full system, FlashAttention scripts included, on its own.)

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-2-vertical-wins/10-pagedattention-kv-cache-pool
pip install -r requirements.txt
```

```bash
python paged_kv_simulator.py    # no GPU, no ingest.py needed — just config.json
```

## Troubleshooting

- **`AutoConfig.from_pretrained` fails with a connection error**: it needs to reach Hugging Face once to download `Qwen/Qwen3-VL-8B-Instruct`'s `config.json` (a few KB, not the model weights) — check your network, or set `HF_HUB_OFFLINE=0` if you've previously restricted it.
- **Your own naive/paged waste percentages differ from the lecture's**: expected — they depend on `MAX_MODEL_LEN`, `BLOCK_SIZE`, and the length distribution `sample_length` draws from, all editable at the top of the script. The *shape* of the result (naive wastes a lot, paging wastes almost nothing) is the point, not the exact percentage.
- **Want to see this for real, not simulated?** That's [vLLM](https://github.com/vllm-project/vllm), the system that actually implements everything this script simulates — Lecture 14 serves the course model on it directly and load-tests it the way Lecture 03 did the naive server.
- Everything from Lectures 01–09's troubleshooting still applies if you also run those systems in this folder.

---

Previous: [`../09-flashattention/`](../09-flashattention/) · Next: [`../11-gqa-mqa-mla/`](../11-gqa-mqa-mla/)
