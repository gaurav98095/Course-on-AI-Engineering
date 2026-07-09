# Lecture 12 — RoPE, ALiBi, YaRN: Positional Encodings

Self-contained: this folder is a **copy-forward** of `11-gqa-mqa-mla/` plus one new file. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 12 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/12-rope-alibi-yarn.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| 💻 Your laptop | `positional_encodings.py` needs no GPU at all — pure numpy on small vectors |
| ⚡ Lightning AI Studio | Everything else from earlier lectures still needs a GPU |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Lecture 11

| File | What it is |
| --- | --- |
| `positional_encodings.py` | Three real, runnable demonstrations: (1) RoPE's rotation, with a numerical proof that its attention score depends only on relative position, never absolute position, (2) ALiBi's explicit per-head distance-penalty slopes, (3) why RoPE's rotation frequencies carry asymmetric extrapolation risk past the trained context length — the exact gap YaRN's frequency-dependent rescaling closes |

This script needs **no GPU** — pure numpy arithmetic on small vectors, no model download at all. Every number it prints is real, reproducible output, computed using our course model's actual `rope_theta` (5,000,000) and `max_position_embeddings` (262,144) from its real `config.json`.

**A labeling note:** our course model uses M-RoPE (a multimodal extension of RoPE that gives image/video tokens separate temporal/height/width position indices) — the lecture explains standard 1D RoPE fully, since it's the mechanism M-RoPE extends, and is explicit about where the real model's config differs.

(Everything from Lectures 01–11 is copied forward unchanged, so this folder still runs the full system, GQA/MQA/MLA comparisons included, on its own.)

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-2-vertical-wins/12-rope-alibi-yarn
pip install -r requirements.txt
```

```bash
python positional_encodings.py    # no GPU, no download needed at all
```

## Troubleshooting

- **Want to check a different model's extrapolation risk?** Edit `ROPE_THETA` and `MAX_POS` at the top of the script to that model's real `config.json` values (`rope_theta`, `max_position_embeddings`).
- **The relative-position scores in Part 1 don't look like real attention scores**: correct — `q` and `k` are random vectors, not real learned weights. The point is the *invariance property* (same `m-n`, same score), not the specific numbers.
- Everything from Lectures 01–11's troubleshooting still applies if you also run those systems in this folder.

---

Previous: [`../11-gqa-mqa-mla/`](../11-gqa-mqa-mla/) · Next: [`../13-continuous-batching/`](../13-continuous-batching/)
