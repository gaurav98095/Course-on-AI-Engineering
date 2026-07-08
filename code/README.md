# Course code, module by module, lecture by lecture

Two levels of folders, both **self-contained**: clone the repo, `cd` into the lecture you're studying, and everything runs from there — data included.

```
code/
  module-1-foundations/
    01-build-a-multimodal-rag/       <- start here
    01b-gpu-vitals/
    02-deploy-it-on-a-gpu/
    03-load-test-it-until-it-breaks/
    04-the-gpu-architecture-and-roofline/
    05-prefill-decode-and-the-kv-cache/
    06-profiling-where-the-time-actually-goes/
  module-2-vertical-wins/
    07-quantization-i-number-formats/
    ...
  module-3-scale/                    (coming)
    20-serving-multimodal-models/
    ...
```

| Module folder | Lectures | The system's state |
| --- | --- | --- |
| `module-1-foundations/` | 01–06 (+ 01b) | Stock Hugging Face code: build the multimodal RAG, serve it with LitServe, load-test it, measure the baseline; GPU vitals and a `/metrics` route woven throughout |
| `module-2-vertical-wins/` | 07–19 | 10× faster: quantization, FlashAttention, vLLM/SGLang, Triton kernels |
| `module-3-scale/` *(coming)* | 20–28 | Ships for 1L+ users: Docker, EKS, autoscaling, SLOs |

## The copy-forward convention

Every new lecture folder **starts as a verbatim copy of the previous lecture's folder** and then adds or changes what that lecture teaches — at both the module level (a new module copies the last module's final lecture) and the lecture level (a new lecture within a module copies the previous lecture in that module). That means:

- Any lecture folder runs standalone from a fresh clone — no need to read any other folder first.
- `diff -rq` between two lecture folders shows *exactly* what that lecture changed.
- Nothing you learned breaks: an old folder keeps working exactly as it did when you built it.
- Each folder's numbers match the benchmark quoted in its lecture.

This mirrors how the corporate migration actually feels: you don't refactor the running system in place — you branch it, transform it, and prove the new one wins with the same benchmark.
