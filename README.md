# Course on AI Engineering

**[Read the course →](https://gaurav98095.github.io/Course-on-AI-Engineering/)**

An advanced, hands-on course on shipping AI systems the way a real engineering team does. In Lecture 01, you build a real multimodal RAG. Then you deploy it, load-test it until it breaks, and spend the rest of the course making it 10× cheaper and faster, then scaling it to a lakh (100,000+) concurrent users.

This is not a slide-deck course. Every lecture is a **build log with numbers** — real code, run on a real GPU, with a real benchmark table that later lectures are judged against. If a lecture claims a speedup, that claim comes with the hardware, precision, and batch size it was measured under, or it's explicitly labeled a ballpark and you're told to go measure your own.

## How this repo is organized

- **[`docs/`](docs/)** — the published course site (GitHub Pages). `docs/lectures/` holds the lectures in reading order; `docs/math/` holds standalone "math deep dive" pages that lectures link out to when a topic has a real derivation worth doing properly, without breaking the hands-on flow.
- **[`code/`](code/)** — one folder per course module, one subfolder per lecture inside it. Every lecture's folder is **fully self-contained**: clone the repo, `cd` into whichever lecture you're on, and everything runs — data included, no need to have read any earlier lecture's folder first. See [`code/README.md`](code/README.md) for the copy-forward convention that makes this work.

## Where you'll run things

| Environment | What it's for |
| --- | --- |
| 💻 Your laptop | Reading the course, clicking around cloud consoles — no GPU code runs here |
| ⚡ [Lightning AI](https://lightning.ai) Studio | Almost every lecture in Modules 1–2 — a cloud GPU dev machine with a browser terminal |
| ☁️ AWS (EC2 / EKS) | Module 3, once the course moves from "build it" to "run it in production" |

Every command block in every lecture is labeled with which of these it runs in — you should never have to guess.

## Hardware & cost

Modules 1–2 are built and tested against one **NVIDIA L40S (48 GB)** on Lightning AI; a 24 GB L4 works for most lectures, tighter on memory. You do not need your own GPU — everything here rents one by the hour. Lightning AI's free tier includes GPU credits to get started; beyond that, expect roughly **$1–2/GPU-hour** on an L40S-class card. No lecture requires more than a few hours of GPU time.

## What you'll be able to do

- Ship a real multimodal RAG, deploy it on a GPU, and load-test it until it breaks.
- Read every benchmark number through GPU and transformer internals — HBM, roofline, prefill vs. decode, KV cache.
- Cut cost per token 10×+ with quantization, FlashAttention, PagedAttention, and continuous batching — and prove quality held, not just assume it.
- Serve on real engines (vLLM, SGLang, TensorRT-LLM) and write Triton kernels underneath them.
- Serve multimodal, embedding, MoE, and reranker models; fine-tune your own embeddings.
- Architect for a lakh concurrent users on Kubernetes with autoscaling, SLOs, and honest capacity math.

## Status

This course is being written module by module, in public. Everything in `docs/lectures/` and `docs/math/` is published and current; the full syllabus (28 lectures across 3 modules) lives on the [course site's home page](https://gaurav98095.github.io/Course-on-AI-Engineering/), with unpublished lectures marked "soon."

## Contributing

This is primarily an authored course, not an open collaborative project — but if you find a bug in the code, a broken link, or a factual error, please [open an issue](https://github.com/gaurav98095/Course-on-AI-Engineering/issues).
