# Lecture 11 — GQA, MQA, MLA: Cheaper Attention Heads

Self-contained: this folder is a **copy-forward** of `10-pagedattention-kv-cache-pool/` plus one new file. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 11 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/11-gqa-mqa-mla.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| 💻 Your laptop | `attention_head_designs.py` needs no GPU at all — pure tensor-shape mechanics and arithmetic |
| ⚡ Lightning AI Studio | Everything else from earlier lectures still needs a GPU |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Lecture 10

| File | What it is |
| --- | --- |
| `attention_head_designs.py` | Two real, runnable demonstrations: (1) `repeat_kv`, reproduced verbatim from `transformers`, showing exactly what GQA stores in cache versus what it computes with, (2) KV-cache-per-token formulas for MHA, GQA, MQA, and MLA, computed for our course model's real dimensions |

This script needs **no GPU** — it's pure PyTorch tensor-shape mechanics plus arithmetic, no model download at all. Every number it prints is real, reproducible output — run it on your laptop.

**A labeling note carried through the script and the lecture:** our course model (Qwen3-VL-8B-Instruct) actually uses GQA — that row is a measured fact from its real `config.json`. The MHA, MQA, and MLA rows are *hypothetical*, computed with the same formula family applied to our model's dimensions, clearly labeled as such — this repo doesn't run those in the model, because it doesn't use them.

(Everything from Lectures 01–10 is copied forward unchanged, so this folder still runs the full system, PagedAttention simulator included, on its own.)

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-2-vertical-wins/11-gqa-mqa-mla
pip install -r requirements.txt
```

```bash
python attention_head_designs.py    # no GPU, no download needed at all
```

## Troubleshooting

- **The cache numbers don't match a different model you're curious about**: edit `L, N_H, D_H, S` and `N_G_ACTUAL` at the top of `cache_comparison()` to that model's real `config.json` values (`num_hidden_layers`, `num_attention_heads`, `head_dim`, `num_key_value_heads`).
- **Want to see MLA for real, not hypothetically?** DeepSeek-V2/V3's own weights and technical report are the real thing this script's MLA row estimates from published ratios — this repo's course model doesn't use MLA, so there's nothing to run here.
- Everything from Lectures 01–10's troubleshooting still applies if you also run those systems in this folder.

---

Previous: [`../10-pagedattention-kv-cache-pool/`](../10-pagedattention-kv-cache-pool/) · This is the last lecture published in Module 2 so far — more lectures land here as the course continues.
