# Lecture 09 — FlashAttention

Self-contained: this folder is a **copy-forward** of `08b-build-the-eval-harness/` plus two new files. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 09 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/09-flashattention.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| ⚡ Lightning AI Studio | Everything — both scripts need a GPU |
| 💻 Your laptop | Browser only, reading the lecture |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Lecture 08b

| File | What it is |
| --- | --- |
| `flash_attention.py` | Standalone micro-benchmark: naive attention (materializes the full N×N score matrix) vs `torch.nn.functional.scaled_dot_product_attention` (PyTorch's fused kernel), swept across sequence length 512→8192, memory and time both measured |
| `attn_backend_compare.py` | Reloads the actual course model with `attn_implementation="eager"` / `"sdpa"` / `"flash_attention_2"` and re-times Lecture 01's own stall question's TTFT |

Neither script needs a new package for `eager`/`sdpa` — PyTorch's own fused kernels ship inside `torch`. Only `attn_backend_compare.py --impl flash_attention_2` needs the standalone `flash-attn` package (commented out in `requirements.txt` — see Troubleshooting).

(Everything from Lectures 01–08b is copied forward unchanged, so this folder still runs the full system, eval harness included, on its own.)

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-2-vertical-wins/09-flashattention
pip install -r requirements.txt
python ingest.py                    # build corpus/ if you don't have it yet
```

```bash
python flash_attention.py                             # no model download needed — pure attention micro-benchmark
python attn_backend_compare.py --impl eager
python attn_backend_compare.py --impl sdpa
```

Optional, needs the standalone package (see Troubleshooting first):

```bash
pip install flash-attn --no-build-isolation
python attn_backend_compare.py --impl flash_attention_2
```

## Troubleshooting

- **`flash_attention.py` prints `OOM` for naive attention at long sequence lengths**: that's the point, not a bug — a 8192-token, 32-head, fp16 score matrix is several GiB on its own. Flash/SDPA never materializes it and keeps running. Note the sequence length where naive first breaks; that's a real number, not a ballpark.
- **`pip install flash-attn` takes 10-20+ minutes or fails**: it compiles CUDA kernels from source against your exact torch/CUDA version — this is normal, not a sign anything is wrong. If it fails, `--impl sdpa` already gets you a fused, flash-attention-style kernel on most modern GPUs without this package at all; treat `flash_attention_2` as an optional, more explicit path, not a requirement for this lecture's core lesson.
- **`RuntimeError: No available kernel` from SDPA**: usually an old PyTorch build with limited backend coverage for your GPU/dtype combination — try `pip install -U torch` first.
- **`attn_backend_compare.py` numbers barely differ between `eager` and `sdpa`**: expected on a short prompt — the fused kernel's advantage grows with sequence length (this is exactly `flash_attention.py`'s point). Try a longer question or a bigger `k_text` in `rag.py`'s `Retriever` call to lengthen the prompt.
- Everything from Lectures 01–08b's troubleshooting still applies if you also run those systems in this folder.

---

Previous: [`../08b-build-the-eval-harness/`](../08b-build-the-eval-harness/) · Next: [`../10-pagedattention-kv-cache-pool/`](../10-pagedattention-kv-cache-pool/)
