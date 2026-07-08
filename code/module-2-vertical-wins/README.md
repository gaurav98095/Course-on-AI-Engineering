# Module 2 · Vertical wins — make it 10× faster

One folder per lecture, same convention as Module 1: each is **fully self-contained** — data included, copy-forward from the previous lecture (the first lecture here copy-forwards from Module 1's final lecture folder).

| Folder | Lecture | What it adds |
| --- | --- | --- |
| [`07-quantization-i-number-formats/`](07-quantization-i-number-formats/) | [07](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/07-quantization-i-number-formats.html) | Number formats (bit-level demo), int8/int4 loading via `bitsandbytes`, real memory + speed measurement |

Diff any two folders — including across the Module 1/2 boundary — to see exactly what a lecture changed:

```bash
diff -rq ../module-1-foundations/06-profiling-where-the-time-actually-goes 07-quantization-i-number-formats
```

More lectures land here as Module 2 continues (GPTQ/AWQ, FlashAttention, PagedAttention, GQA/MQA/MLA, RoPE/ALiBi/YaRN, continuous batching, vLLM/SGLang, TensorRT-LLM, speculative decoding, prefix/chunked prefill, Triton kernels, torch.compile/CUDA graphs — Lectures 08–19).
