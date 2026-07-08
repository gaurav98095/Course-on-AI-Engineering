# Module 1 · Foundations — build it, deploy it, break it

One folder per lecture. Each lecture folder is **fully self-contained** — data included — so you can clone the repo and start at whichever lecture you're on without reading any other folder first.

| Folder | Lecture | What it adds |
| --- | --- | --- |
| [`01-build-a-multimodal-rag/`](01-build-a-multimodal-rag/) | [01](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/01-build-a-multimodal-rag.html) | The system: ingest, embed, index, retrieve, generate |
| [`02-deploy-it-on-a-gpu/`](02-deploy-it-on-a-gpu/) | [02](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/02-deploy-it-on-a-gpu.html) | + a LitServe HTTP endpoint |
| [`03-load-test-it-until-it-breaks/`](03-load-test-it-until-it-breaks/) | [03](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/03-load-test-it-until-it-breaks.html) | + a closed-loop load generator |
| [`04-the-gpu-architecture-and-roofline/`](04-the-gpu-architecture-and-roofline/) | [04](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/04-the-gpu-architecture-and-roofline.html) | + a standalone GPU roofline benchmark |

Each folder is a **copy-forward** of the previous one — same convention the whole course uses, applied per lecture within the module (see [`../README.md`](../README.md) for the module-level version). Diff any two folders to see exactly what that lecture changed:

```bash
diff -rq 03-load-test-it-until-it-breaks 04-the-gpu-architecture-and-roofline
```

More lectures land here as Module 1 continues (KV cache math, profiling).
