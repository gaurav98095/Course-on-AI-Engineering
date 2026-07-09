---
layout: default
title: "Math Deep Dive — Deriving the KV Cache Formula"
---

# Math Deep Dive — Deriving the KV Cache Formula

> **Supports:** [Lecture 05 — Prefill, Decode, and the KV Cache](../lectures/05-prefill-decode-and-the-kv-cache.md). Read the lecture first; this page assumes its story and mental model.

## The Idea in One Picture

Self-attention needs, for every token, a key vector and a value vector at every layer, forever — every future token will compare itself against them. The KV cache is simply "don't throw those away and recompute them." Its size is nothing more than counting how many such vectors exist and multiplying by how big each one is.

## Notation

| Symbol | Meaning | Course model's value |
| --- | --- | --- |
| \\(L\\) | number of transformer layers | 36 |
| \\(H\\) | number of KV heads (not query heads — see Part 2) | 8 |
| \\(H_q\\) | number of query heads | 32 |
| \\(D\\) | dimension of one head | 128 |
| \\(t\\) | tokens cached (prompt + generated so far) | varies |
| \\(b\\) | batch size (concurrent sequences) | varies |
| \\(s\\) | bytes per stored number | 2 (bf16/fp16), 1 (fp8) |

## Derivation

### Part 1 — Counting the vectors, then their bytes

Self-attention at one layer produces, for every input token, one key vector \\(k \in \mathbb{R}^D\\) and one value vector \\(v \in \mathbb{R}^D\\) *per KV head*. To answer "what does token 500 need to attend to token 3," the model needs token 3's key and value **available**, at **that layer**, forever — every later token will query against it.

Count the vectors that must be kept, for a sequence of \\(t\\) tokens, batch \\(b\\), across all \\(L\\) layers and \\(H\\) KV heads:

\\[
\#\text{vectors} = \underbrace{2}_{\text{K and V}} \times L \times H \times t \times b
\\]

Each vector has \\(D\\) numbers, each stored in \\(s\\) bytes. Multiply through:

\\[
\boxed{\text{cache bytes} = 2 \, L \, H \, D \, t \, b \, s}
\\]

Six factors, six independent levers — and every one of them enters **linearly**. Double any single factor and the cache exactly doubles. There is no economy of scale hiding anywhere in this formula; that's precisely why it can blow past any GPU's VRAM so easily (Lecture 05's 288 GiB row).

### Part 2 — Why \\(H\\) is the *KV* head count, not the query head count

A transformer computes \\(H_q\\) separate query heads, but a model using **grouped-query attention (GQA)** computes far fewer *key and value* heads — several query heads share one KV head, computing attention against the same cached keys and values. Our course model has \\(H_q = 32\\) query heads sharing just \\(H=8\\) KV heads: a group size of 4.

The cache only ever stores keys and values, never queries — a query is computed fresh for the current token and discarded immediately after producing that token's attention output. So the cache formula uses \\(H\\), the KV head count, not \\(H_q\\). This is not a simplification; it's the literal mechanism. If our model instead computed one KV head per query head (\\(H = H_q = 32\\), "multi-head attention," the pre-GQA default), the cache would be exactly \\(H_q / H = 4\times\\) larger for identical model quality on the query side — Exercise 3 in the lecture asks you to verify this by substitution.

### Part 3 — Where the terms come from, mapped onto real memory

It's worth walking the multiplication out loud once, because each factor corresponds to something you can point at:

- \\(2\\): one array for keys, a separate array for values — they're never merged.
- \\(L\\): every layer keeps its own cache. A 36-layer model doesn't share a cache across layers; layer 1's keys are meaningless to layer 2's attention.
- \\(H\\): heads run in parallel over disjoint slices of the representation; each needs its own stored vectors.
- \\(D\\): the size of one head's slice — how much of the representation one head "sees."
- \\(t\\): grows by exactly one, once, every decode step — this is *the* variable that changes during a request.
- \\(b\\): completely independent sequences stored side by side; no sharing across requests (barring prefix caching, Lecture 17).
- \\(s\\): the only factor precision-reduction techniques touch directly — Lecture 07 quantizes *weights*; Lecture 10 (PagedAttention & the KV Cache Pool) covers quantizing the *cache* itself, the same lever applied to a different tensor.

### Part 4 — The two knobs that trade against each other

Fix everything except \\(t\\) and \\(b\\); their product alone determines cache size:

\\[
\text{cache bytes} \propto t \times b
\\]

This is the algebra behind the lecture's table: \\((t{=}4096, b{=}8)\\) and \\((t{=}32768, b{=}1)\\) give the *identical* cache size, because \\(4096 \times 8 = 32768 \times 1\\). A serving system doesn't get to choose "long context" and "high concurrency" independently — it's spending the same budget on a single product, and every serving-engine feature in Module 2 (paged allocation, prefix caching, quantized cache) exists to relax this one constraint.

## Worked Example

Course model: \\(L{=}36\\), \\(H{=}8\\), \\(D{=}128\\), bf16 \\(s{=}2\\).

**Per token, one layer:** \\(2 \times 8 \times 128 \times 2 = 4{,}096\\) bytes = 4 KiB.

**Per token, all layers:** \\(4{,}096 \times 36 = 147{,}456\\) bytes = **144 KiB** — this single number is the multiplier behind every row of the lecture's table.

**Our RAG's real prompt** (\\(t{=}1{,}800\\), \\(b{=}1\\)): \\(147{,}456 \times 1{,}800 = 265{,}420{,}800\\) bytes \\(\approx 253\\) MiB.

**A serving load** (\\(t{=}4{,}096\\), \\(b{=}8\\)): \\(147{,}456 \times 4{,}096 \times 8 \approx 4.50\\) GiB — nearly a tenth of an L40S's 48 GiB, spent on cache alone, before a single weight is counted.

**The breaking point** (\\(t{=}32{,}768\\), \\(b{=}64\\)): \\(147{,}456 \times 32{,}768 \times 64 \approx 288\\) GiB — no GPU, and no realistic multi-GPU node, holds this in KV cache while also holding the model.

**Without GQA** (using \\(H_q{=}32\\) instead of \\(H{=}8\\)): every one of the numbers above would be exactly \\(4\times\\) larger. GQA is, quite literally, the difference between the 4.50 GiB row and an 18 GiB row.

## Where the Assumptions Break

**Not every architecture caches this way.** Some newer designs share a cache across groups of layers, or compress the cache with a learned projection (a Module 2/3-adjacent research direction). The formula here is the standard, dense, per-layer cache every mainstream open model (including ours) actually uses — treat it as the default, not a universal law.

**Real allocators don't hit the formula exactly.** Naively allocating one contiguous block per sequence, sized for the maximum possible length, wastes enormous amounts of memory on sequences that end early — this is precisely the fragmentation problem PagedAttention (Lecture 10) solves. Our formula computes the *cache content's* size; it says nothing about how much memory a naive *allocator* reserves to hold it.

**Precision changes \\(s\\), and \\(s\\) alone.** Quantizing the KV cache to fp8 halves every number in this page's worked example without changing \\(L\\), \\(H\\), \\(D\\), \\(t\\), or \\(b\\) at all — the cleanest possible lever, and one of the cheapest wins in the entire course. Lecture 07 builds the number-format vocabulary this depends on; Lecture 10 applies it to the cache specifically, alongside PagedAttention's separate fix for how the cache is *allocated*.

**Multi-request batching only helps throughput if the cache fits.** Continuous batching (Lecture 13) wants \\(b\\) as large as possible to keep arithmetic intensity high (recall Lecture 04's Part 4: \\(\text{AI} = 2b/s\\) for a decode step batched across \\(b\\) requests). But this page's formula is the hard ceiling on how large \\(b\\) can go before the cache alone exhausts VRAM — the two lectures are two sides of the same coin.

## Common Mistakes

- **Using query head count instead of KV head count.** On any model with GQA or MQA (nearly every modern open model), this overestimates the cache by the group-size factor — check `num_key_value_heads`, never `num_attention_heads`, in a config.
- **Forgetting the factor of 2.** Keys and values are separate arrays; a common slip is computing only one and forgetting to double it.
- **Treating cache size as fixed per request.** It grows by one token's worth *every single decode step* — a formula evaluated once at request start is already stale by the second generated token. Real serving systems track this growth live.
- **Ignoring batch when reasoning about "will this fit."** A model comfortably serving one long-context request can still run out of memory the moment a second concurrent request arrives — \\(b\\) is not a free variable, it's part of the same product as \\(t\\).

---

[← Back to Lecture 05 — Prefill, Decode, and the KV Cache](../lectures/05-prefill-decode-and-the-kv-cache.md) · [Course Home](../index.md)
