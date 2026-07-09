---
layout: default
title: "Math Deep Dive — The Arithmetic of Paged Memory"
---

# Math Deep Dive — The Arithmetic of Paged Memory

> **Supports:** [Lecture 10 — PagedAttention & the KV Cache Pool](../lectures/10-pagedattention-kv-cache-pool.md). Read the lecture first; this page assumes its naive-vs-paged simulation and its self-storage mental model.

## The Idea in One Picture

Two different kinds of waste, two different closed-form expectations. Naive allocation wastes whatever a request doesn't use of its worst-case reservation — a large, systematic loss driven by how far the *average* request falls short of the *maximum* one. Paged allocation wastes at most one partially-filled block per request — a small, structural loss driven by simple probability, the same kind of reasoning Math Deep Dive 08b used for eval-set uncertainty. This page derives both, plus the asymptotic payoff of sharing a prefix across many requests.

## Notation

| Symbol | Meaning | Lecture 10's simulation |
| --- | --- | --- |
| \\(M\\) | max context length — what naive allocation must reserve per request | 8192 |
| \\(B\\) | block size — paged allocation's fixed unit | 16 |
| \\(\ell\\) | one request's actual length (a random variable) | mean 907 |
| \\(E[\ell]\\) | expected request length | 907.2 (empirical) |
| \\(N\\) | number of requests | 1,000 (fragmentation demo), 50 (sharing demo) |

## Derivation

### Part 1 — Naive waste: how far short of the max the average request falls

Naive allocation reserves \\(M\\) tokens for every request, regardless of how many it uses. Total reserved is \\(NM\\); total used is \\(\sum \ell\_i \approx N\, E[\ell]\\) for large \\(N\\). The fraction wasted:

\\[
\text{naive waste} = \frac{NM - N\,E[\ell]}{NM} = 1 - \frac{E[\ell]}{M}
\\]

This is not a small-sample estimate — it is the exact fraction, in the limit of many requests, and it only depends on one ratio: how big the *average* request is relative to the *worst-case* one the system must plan for. A system whose typical request is 10% of its max context wastes 90%, no matter how efficient anything else about it is.

### Part 2 — Paged waste: the last block is the only place waste can hide

Paged allocation rounds each request's reservation up to the next multiple of \\(B\\). The only wasted space is the gap between a request's actual length and that rounding — at most \\(B-1\\) tokens, never a whole extra block. Write the remainder as \\(r = \ell \bmod B\\); the waste for that one request is \\((B - r) \bmod B\\).

If a request's length is effectively unrelated to the block size — true for any real traffic mix, since nothing about token counts is naturally a multiple of 16 — the remainder \\(r\\) is well-approximated as **uniformly distributed** over \\(\{0, 1, \ldots, B-1\}\\). The expected waste per request under a uniform remainder:

\\[
E[\text{waste per request}] = \frac{1}{B}\sum\_{r=0}^{B-1}(B-r) \bmod B \cdot [\text{r>0}] \approx \frac{B-1}{2}
\\]

(The exact sum: for \\(r=0\\) waste is 0; for \\(r=1,\ldots,B-1\\) waste is \\(B-r\\). Averaging \\(B-1, B-2, \ldots, 1\\) over \\(B\\) outcomes gives \\(\frac{(B-1)B/2}{B} = \frac{B-1}{2}\\).) Divide by the average request size to get a *fraction* wasted:

\\[
\text{paged waste} \approx \frac{(B-1)/2}{E[\ell]}
\\]

Unlike naive waste, this shrinks as requests get *longer* (a bigger \\(\ell\\) makes the fixed half-block loss a smaller fraction of it) and shrinks as blocks get *smaller* (less to lose in the worst case) — the opposite dependence from naive waste, which doesn't care about \\(B\\) at all.

### Part 3 — Prefix sharing: the savings has a clean limit

\\(N\\) requests, each \\(P\\) shared prefix tokens plus \\(U\\) unique tokens. Without sharing, total storage is \\(N(P+U)\\); with sharing, the prefix is stored once: \\(P + NU\\).

\\[
\text{savings} = 1 - \frac{P + NU}{N(P+U)}
\\]

As \\(N\\) grows, the shared prefix's one-time cost \\(P\\) gets divided across more and more requests and its per-request contribution vanishes:

\\[
\lim\_{N\to\infty} \text{savings} = 1 - \frac{U}{P+U} = \frac{P}{P+U}
\\]

The asymptotic savings ceiling depends **only on the ratio of shared to unique tokens** — not on \\(N\\) at all, once \\(N\\) is large enough. A system whose shared prefix is 10× its average unique suffix eventually saves close to \\(10/11 \approx 91\%\\), regardless of whether it's serving 100 requests or 100,000.

### Part 4 — Stacking independent levers

Paging changes *how much* gets reserved; quantizing the cache (Lecture 07's lever) changes *how many bytes per token* that reservation costs. Neither formula above has any dependence on the other — they multiply:

\\[
\text{total memory} = (\text{tokens reserved, from Part 1 or 2}) \times (\text{bytes per token, from Lecture 05's formula})
\\]

Halve the bytes per token and the memory halves, at *any* fixed number of tokens reserved — this is exactly why Lecture 10's Step 4 shows the same 2× ratio (bf16 → int8) regardless of whether paging is already applied.

## Worked Example

Lecture 10's own simulation, checked against every formula above. \\(E[\ell] = 907.2\\), \\(M=8192\\), \\(B=16\\):

**Naive waste:** \\(1 - 907.2/8192 = 1 - 0.1108 = 0.8892\\) — predicted 88.9%, simulated 88.9%.

**Paged waste:** \\((16-1)/2 = 7.5\\) expected tokens lost per request; \\(7.5 / 907.2 = 0.00827\\) — predicted 0.83%, simulated 0.8%.

**Prefix sharing**, \\(N=50\\), \\(P=500\\), \\(U=50\\): exact savings \\(=1 - \frac{500+50\times50}{50\times550} = 1 - \frac{3000}{27500}=0.891\\) — 89.1%, matching the simulation exactly (this one has no approximation — Part 3's first formula is exact at any \\(N\\), the *limit* in Part 3 is what's approximate for finite \\(N\\)). The asymptotic ceiling as \\(N\to\infty\\) would be \\(500/550 = 90.9\\%\\) — already almost reached at \\(N=50\\).

## Where the Assumptions Break

**The uniform-remainder assumption can fail.** If request lengths cluster near multiples of the block size (unlikely for natural language token counts, but possible for synthetic or padded workloads), the true expected waste can differ meaningfully from \\((B-1)/2\\). Always check a real system's actual length distribution before trusting the approximation blindly — Lecture 10's Exercise 2 asks you to verify it against your own traffic mix.

**Part 1 and Part 2 ignore per-request bookkeeping.** A real system's block table itself costs a small amount of memory per request, and Part 2's formula only counts wasted *cache* tokens, not the pointers tracking them. This overhead is real but tiny relative to the cache itself (a handful of integers per request vs. hundreds of KiB of cache), so it doesn't change either conclusion qualitatively.

**Part 3 assumes the shared prefix is truly identical, token for token, at block boundaries.** One differing token anywhere in the shared region breaks sharing for that block (Lecture 10's own Where It Breaks) — the formula computes the *ceiling* achievable with perfect sharing, not what a fuzzier real-world prefix match would deliver.

## Common Mistakes

- **Assuming paged waste depends on \\(M\\).** It doesn't — Part 2's formula has no \\(M\\) term at all. Naive waste is a story about the gap between average and worst case; paged waste is a story about rounding to a block boundary, a completely different mechanism.
- **Forgetting prefix-sharing savings has a ceiling.** Adding more requests beyond the point where \\(N\\) is already "large" (relative to \\(P/U\\)) barely moves the savings percentage further — the asymptote in Part 3 is real, not just a simplification.
- **Multiplying waste percentages instead of the underlying token counts.** "88.9% waste plus another 50% reduction from quantization" is not "38.9% total waste" — always work in the actual reserved-token or reserved-byte numbers (Part 4), then convert to a percentage at the end.

---

[← Back to Lecture 10 — PagedAttention & the KV Cache Pool](../lectures/10-pagedattention-kv-cache-pool.md) · [Course Home](../index.md)
