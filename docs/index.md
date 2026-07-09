---
layout: default
title: "Course on AI Engineering"
---

# Course on AI Engineering

An advanced, hands-on course on shipping AI systems the way a real engineering team does. We build a multimodal RAG in week one, deploy it on a GPU, and load-test it until it breaks — then spend the rest of the course making it 10× cheaper and 10× faster, and scaling it to a lakh concurrent users. Quantization, FlashAttention, PagedAttention, vLLM, SGLang, TensorRT-LLM, Triton kernels, speculative decoding, Kubernetes, SLOs, and the capacity math behind $/million tokens — every claim measured, every number explained.

## How to use this course

Read each lecture in order and run the code — every hands-on lecture has a `code/` folder that runs standalone. Weeks 1–8 run on a single Lightning AI GPU Studio; the scale module moves to AWS. Heavy derivations live in separate math deep-dive pages, linked from each lecture, so the build never stalls.

You should be comfortable with Python and the deep-learning basics (backprop, CNNs/RNNs, self-attention). Everything past that, we build from scratch.

## Module 1 · Foundations — build it & measure it

Ship a real multimodal RAG on stock code, deploy it, and load-test it until it breaks — then learn the GPU and transformer fundamentals that explain every number in your benchmark.

<ol class="lectures">
  <li>
    <a href="lectures/01-build-a-multimodal-rag.html">
      <span class="num">01</span>
      <span>
        <span class="t">Build a Multimodal RAG</span>
        <span class="d" style="display:block">Text + image retrieval over a real 500-page illustrated manual, answered by a vision-language model — in stock Hugging Face code.</span>
      </span>
      <span class="go">→</span>
    </a>
  </li>
  <li>
    <a href="lectures/01b-gpu-vitals.html">
      <span class="num">01b</span>
      <span>
        <span class="t">GPU Vitals: Watching What You Built</span>
        <span class="d" style="display:block">Stop glancing at nvidia-smi once — watch utilization, memory, power, and temperature continuously, before we deploy anything.</span>
      </span>
      <span class="go">→</span>
    </a>
  </li>
  <li>
    <a href="lectures/02-deploy-it-on-a-gpu.html">
      <span class="num">02</span>
      <span>
        <span class="t">Deploy It on a GPU</span>
        <span class="d" style="display:block">From script to service: a LitServe endpoint with a schema, health check, public URL, a /metrics route — and one process quietly playing two different roles.</span>
      </span>
      <span class="go">→</span>
    </a>
  </li>
  <li>
    <a href="lectures/03-load-test-it-until-it-breaks.html">
      <span class="num">03</span>
      <span>
        <span class="t">Load-Test It Until It Breaks</span>
        <span class="d" style="display:block">Concurrency 1 → 32: throughput pins at a ceiling, latency climbs Little's law, requests die at the timeout cliff — and the GPU turns out to be &lt;1% used.</span>
      </span>
      <span class="go">→</span>
    </a>
  </li>
  <li>
    <a href="lectures/04-the-gpu-architecture-and-roofline.html">
      <span class="num">04</span>
      <span>
        <span class="t">The GPU: Architecture, HBM, and the Roofline Model</span>
        <span class="d" style="display:block">SMs, HBM, and one plot that predicts — and explains Lecture 03's &lt;1% GPU utilization — whether any workload is compute-bound or memory-bound.</span>
      </span>
      <span class="go">→</span>
    </a>
  </li>
  <li>
    <a href="lectures/05-prefill-decode-and-the-kv-cache.html">
      <span class="num">05</span>
      <span>
        <span class="t">Prefill, Decode, and the KV Cache</span>
        <span class="d" style="display:block">Why the first token and every token after it are two different workloads — and the cache math that rules serving.</span>
      </span>
      <span class="go">→</span>
    </a>
  </li>
  <li>
    <a href="lectures/06-profiling-where-the-time-actually-goes.html">
      <span class="num">06</span>
      <span>
        <span class="t">Profiling: Where the Time Actually Goes</span>
        <span class="d" style="display:block">torch.profiler, nsys, and ncu on our own endpoint: attach names to every millisecond, and check the roofline against the vendor's own counters.</span>
      </span>
      <span class="go">→</span>
    </a>
  </li>
</ol>

**Module 1 is complete.** Module 2 begins below.

## Module 2 · Vertical wins — make it 10× faster

Where most of the 10× cost reduction comes from. Shrink the model, speed up attention, serve it on a real engine, write the kernels underneath yourself — then prove the quality held.

<ol class="lectures">
  <li>
    <a href="lectures/07-quantization-i-number-formats.html">
      <span class="num">07</span>
      <span>
        <span class="t">Quantization I — Number Formats</span>
        <span class="d" style="display:block">BF16, FP16, FP8, INT8, INT4: what precision buys, what it costs — and why quantized memory is free but quantized speed isn't.</span>
      </span>
      <span class="go">→</span>
    </a>
  </li>
  <li>
    <a href="lectures/08-quantization-ii-gptq-and-awq.html">
      <span class="num">08</span>
      <span>
        <span class="t">Quantization II — GPTQ &amp; AWQ in Practice</span>
        <span class="d" style="display:block">Calibrated quantization with real kernels: the speedup Lecture 07 promised, plus a quality check to prove the answers survived.</span>
      </span>
      <span class="go">→</span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">08b</span>
      <span>
        <span class="t">Build the Eval Harness</span>
        <span class="d" style="display:block">Turn "the answer looked right" into a real, reusable score — recall@k plus answer grading, re-run after every optimization from here on.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">09</span>
      <span>
        <span class="t">FlashAttention</span>
        <span class="d" style="display:block">The same math, tiled to never touch HBM twice — the kernel that made long context affordable.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">10</span>
      <span>
        <span class="t">PagedAttention &amp; the KV Cache Pool</span>
        <span class="d" style="display:block">Virtual memory for the KV cache: how vLLM stopped wasting 60% of your VRAM — plus quantizing the cache itself, the same lever from Lecture 07 applied to a different tensor.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">11</span>
      <span>
        <span class="t">GQA, MQA, MLA — Cheaper Attention Heads</span>
        <span class="d" style="display:block">Share the keys and values, keep the quality: the head designs behind every modern model.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">12</span>
      <span>
        <span class="t">Positional Encodings — RoPE, ALiBi, YaRN</span>
        <span class="d" style="display:block">How models know where a token is, and how to stretch context past what they trained on.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">13</span>
      <span>
        <span class="t">Continuous Batching</span>
        <span class="d" style="display:block">Seat every guest the moment a chair frees up: the scheduling trick worth more than any kernel.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">14</span>
      <span>
        <span class="t">vLLM &amp; SGLang</span>
        <span class="d" style="display:block">Serve the course model on real engines, benchmark them head-to-head, and read the flags that matter.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">15</span>
      <span>
        <span class="t">TensorRT-LLM</span>
        <span class="d" style="display:block">Compile the model for the exact GPU it runs on — and see when the extra effort pays.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">16</span>
      <span>
        <span class="t">Speculative Decoding</span>
        <span class="d" style="display:block">A small model drafts, the big model verifies — identical outputs, twice the speed.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">17</span>
      <span>
        <span class="t">Prefix Caching &amp; Chunked Prefill</span>
        <span class="d" style="display:block">Stop re-reading the system prompt: cache hits measured in dollars.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">18</span>
      <span>
        <span class="t">Write Your First Triton Kernel</span>
        <span class="d" style="display:block">The kernel under everything: write fused softmax yourself and race it against PyTorch.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">19</span>
      <span>
        <span class="t">torch.compile &amp; CUDA Graphs</span>
        <span class="d" style="display:block">Kill Python overhead and launch latency — the last free milliseconds.</span>
      </span>
    </a>
  </li>
</ol>

## Module 3 · Modalities & scale — ship for 1L+ users

Apply the toolkit to multimodal, embedding and MoE models, finetune your own embeddings, then architect, ship and observe the whole thing for a lakh concurrent users on AWS.

<ol class="lectures">
  <li>
    <a class="soon">
      <span class="num">20</span>
      <span>
        <span class="t">Serving Multimodal Models</span>
        <span class="d" style="display:block">Vision towers, image tokens, and why multimodal prefill is a different beast.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">21</span>
      <span>
        <span class="t">Embedding &amp; Reranker Models at Scale</span>
        <span class="d" style="display:block">The other half of RAG: high-throughput embedding serving and two-stage retrieval.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">22</span>
      <span>
        <span class="t">Mixture of Experts</span>
        <span class="d" style="display:block">Why a 100B-parameter model can cost like a 10B one — and what routing does to serving.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">23</span>
      <span>
        <span class="t">Finetune Your Own Embeddings</span>
        <span class="d" style="display:block">Contrastive training on your own corpus, plus Matryoshka embeddings that shrink on demand.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">24</span>
      <span>
        <span class="t">Disaggregation &amp; Multi-LoRA</span>
        <span class="d" style="display:block">Split prefill from decode across GPUs, and serve a hundred finetunes from one base model.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">25</span>
      <span>
        <span class="t">Docker for GPUs &amp; the Move to AWS</span>
        <span class="d" style="display:block">Containerize the whole stack and land it on EC2 GPU instances — the corporate migration, done right.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">26</span>
      <span>
        <span class="t">Kubernetes (EKS) &amp; Autoscaling</span>
        <span class="d" style="display:block">HPA, KEDA, and GPU-aware scheduling: the cluster that grows with your traffic.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">27</span>
      <span>
        <span class="t">SLOs, p95/p99, and Observability</span>
        <span class="d" style="display:block">Dashboards that catch the fire before users do: latency percentiles, GPU metrics, alerts.</span>
      </span>
    </a>
  </li>
  <li>
    <a class="soon">
      <span class="num">28</span>
      <span>
        <span class="t">$/Mtok &amp; Capacity Math</span>
        <span class="d" style="display:block">The CFO lecture: turn every optimization in this course into a cost model for a lakh users.</span>
      </span>
    </a>
  </li>
</ol>

## Math deep dives

Each hands-on lecture links to a separate page with the full derivations — every symbol defined, every step justified.

- [Math Deep Dive 01 — The Geometry of Retrieval](math/01-geometry-of-retrieval.md) — dot products, cosine similarity, contrastive training, and why one vector space can hold both images and text.
- [Math Deep Dive 01b — Sampling Rate and Smoothing a Live Signal](math/01b-sampling-and-smoothing.md) — why a 1-second poll can miss a 30ms spike entirely, and the exponential moving average formula real dashboards use to smooth noisy signals.
- [Math Deep Dive 03 — Queues, Percentiles, and Why p95 Explodes](math/03-queues-and-percentiles.md) — Little's law from scratch, M/M/1, the 1/(1−ρ) hockey stick, and how load tests lie.
- [Math Deep Dive 04 — Deriving the Roofline Model](math/04-roofline-model.md) — the min(compute, AI×bandwidth) formula, arithmetic intensity of a GEMM, and why batching is the same lever as more tokens.
- [Math Deep Dive 05 — Deriving the KV Cache Formula](math/05-kv-cache-math.md) — counting key/value vectors, why GQA shrinks the cache 4×, and the (tokens × batch) budget every serving system fights over.
- [Math Deep Dive 06 — Amdahl's Law and Where to Spend an Hour](math/06-amdahls-law.md) — the exact formula behind "optimize the biggest bar in the profile, not your favorite line of code."
- [Math Deep Dive 07 — Why Quantized Speed Isn't Free the Way Quantized Memory Is](math/07-quantization-speed-gap.md) — the dequantization-overhead formula that explains why int8 often doesn't speed up decode while int4 usually does.
- [Math Deep Dive 08 — The Compensation Formula Behind GPTQ](math/08-gptq-compensation.md) — deriving the layer-wise reconstruction-error objective and the closed-form update that lets later weights compensate for earlier rounding, with a full worked 2-weight example.

## What you will be able to do

- Ship a real multimodal RAG, deploy it on a GPU, and load-test it until it breaks.
- Read every benchmark number through GPU and transformer internals — HBM, roofline, prefill vs decode, KV cache.
- Cut cost per token 10×+ with quantization, FlashAttention, PagedAttention, and continuous batching — and prove quality held.
- Serve on real engines (vLLM, SGLang, TensorRT-LLM) and write Triton kernels underneath them.
- Serve multimodal, embedding, MoE and reranker models; finetune your own embeddings.
- Architect for a lakh concurrent users on Kubernetes with autoscaling, SLOs, and honest capacity math.

---

Start here: [Lecture 01 — Build a Multimodal RAG →](lectures/01-build-a-multimodal-rag.md)
