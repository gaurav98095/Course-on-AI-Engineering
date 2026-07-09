---
layout: default
title: "Math Deep Dive — Arithmetic Intensity of Grouped Attention"
---

# Math Deep Dive — Arithmetic Intensity of Grouped Attention

> **Supports:** [Lecture 11 — GQA, MQA, MLA: Cheaper Attention Heads](../lectures/11-gqa-mqa-mla.md). Read the lecture first; this page assumes its `repeat_kv` mechanics and its cache-per-token table.

## The Idea in One Picture

Head-sharing doesn't make attention compute less — every query head still produces its own output, so the FLOP count is untouched. What it changes is how many *bytes* have to move from HBM to compute that identical amount of work. That's a roofline-model story, in Lecture 04's exact sense: same FLOPs, fewer bytes, higher arithmetic intensity. This page derives the exact multiplier, and checks whether it's ever enough to make decode attention stop being memory-bound.

## Notation

| Symbol | Meaning | Our course model |
| --- | --- | --- |
| \\(n\_h\\) | number of query heads | 32 |
| \\(n\_g\\) | number of KV groups (1 for MQA, \\(n\_h\\) for MHA) | 8 (real GQA config) |
| \\(N\\) | number of cached tokens attended to | varies per request |
| \\(d\_h\\) | head dimension | 128 |
| \\(s\\) | bytes per element | 2 (bf16) |

## Derivation

### Part 1 — FLOPs don't care how KV heads are shared

One decode step attends one new query against \\(N\\) cached positions. Every one of the \\(n\_h\\) query heads computes its own scores and its own weighted sum over \\(V\\), regardless of whether its K/V came from a private head or a shared group — grouping changes *which* K/V a head reads, not how much work that head does:

\\[
\text{FLOPs} \approx \underbrace{2\,n\_h N d\_h}\_{QK^\top} + \underbrace{2\,n\_h N d\_h}\_{\text{attn} \times V} = 4\,n\_h N d\_h
\\]

This has no \\(n\_g\\) in it at all. Grouping is invisible to the FLOP count.

### Part 2 — Bytes moved scale with groups, not query heads

The bytes read from HBM for this step are exactly the cached K and V for however many *distinct* KV representations exist — \\(n\_g\\) of them, not \\(n\_h\\):

\\[
\text{bytes} = 2\, n\_g N d\_h s
\\]

This is Lecture 05's cache formula for one step's worth of reads, with \\(n\_g\\) in place of what Lecture 05 called \\(H\\) — the same symbol, now written to make the group-count dependence explicit.

### Part 3 — The ratio, and what cancels

\\[
\text{AI} = \frac{\text{FLOPs}}{\text{bytes}} = \frac{4\,n\_h N d\_h}{2\,n\_g N d\_h s} = \frac{2\,n\_h}{n\_g\, s}
\\]

\\(N\\) and \\(d\_h\\) cancel completely — arithmetic intensity depends on **only** the head-count ratio and the precision, not on context length or model width. This mirrors Lecture 04's math page finding that decode's linear-layer arithmetic intensity is exactly 1 FLOP/byte regardless of \\(d\\) — the same structural reason applies here: the terms that make attention expensive to *compute* are exactly the terms that make it expensive to *read*, so they cancel in the ratio.

### Part 4 — Where this lands relative to the ridge point

Plug in \\(s=2\\) (bf16) and our model's \\(n\_h=32\\):

| Design | \\(n\_g\\) | AI (FLOP/byte) | % of L40S ridge point (419, Lecture 04) |
| --- | --- | --- | --- |
| MHA | 32 | 1.0 | 0.2% |
| GQA (ours) | 8 | 4.0 | 1.0% |
| MQA | 1 | 32.0 | 7.6% |

The MHA row's **AI = 1.0 FLOP/byte** exactly matches Lecture 04 and Lecture 05's already-published "decode is ≈1 FLOP/byte, deeply memory-bound" finding — this derivation and that one describe the identical operation from two different angles, and they agree.

Even MQA's full 32× arithmetic-intensity boost only reaches 7.6% of the L40S's ridge point. **Head-sharing moves decode attention linearly closer to compute-bound, but on current hardware, nowhere near far enough to cross the line.** It's a real, useful lever — Lecture 05 already measured its cache-size payoff — but it does not turn decode into a compute-bound operation, at any group size this lecture considered.

## Worked Example

Confirm Part 3's cancellation holds at two wildly different context lengths and head dimensions, not just our own model's numbers — \\(N=100, d\_h=64\\) and \\(N=8000, d\_h=256\\), both with \\(n\_h=32\\), \\(n\_g=8\\), \\(s=2\\):

\\[
\text{AI} = \frac{4(32)(100)(64)}{2(8)(100)(64)(2)} = \frac{819{,}200}{204{,}800} = 4.0
\\]
\\[
\text{AI} = \frac{4(32)(8000)(256)}{2(8)(8000)(256)(2)} = \frac{104{,}857{,}600}{26{,}214{,}400} = 4.0
\\]

Identical answer both times, exactly as Part 3's cancellation predicts — \\(N\\) and \\(d\_h\\) genuinely don't matter, only the 4:1 head-to-group ratio does.

## Where the Assumptions Break

**This counts only the attention operation, not the whole decode step.** A real decode step also runs the linear projections (Q/K/V/O projections, MLP) that Lecture 04's math page already analyzed — those have their own, separate arithmetic intensity (also ≈1 FLOP/byte, for unrelated reasons) and are unaffected by head-sharing at all, since they're weight-read-dominated, not cache-read-dominated. Grouping only moves the needle on the *attention* portion of the step.

**Real kernels don't read exactly Part 2's byte count.** Padding, alignment, and how a specific attention kernel tiles its reads (Lecture 09's FlashAttention territory) all add overhead this idealized count doesn't capture. Treat the AI values here as the same kind of ceiling Lecture 04's roofline always describes — a bound, not a promise every kernel hits it.

**Prefill is a different regime entirely.** This whole derivation is decode-specific (\\(N\\) cached, one new query). Lecture 05 already established prefill is compute-bound for an unrelated reason (all \\(N\\) query positions computed together) — head-sharing's arithmetic-intensity argument doesn't apply there the same way.

## Common Mistakes

- **Assuming GQA reduces attention FLOPs.** It doesn't — Part 1 has no \\(n\_g\\) term. Every query head still does full-price attention math; only the *memory traffic* for K/V shrinks.
- **Treating a higher AI as "solved the memory-bound problem."** Part 4's table is the correction: even MQA's 32× boost lands at single-digit percent of the ridge point. Grouping is a real, worthwhile lever — it is not a fix for decode's fundamental memory-bound nature.
- **Forgetting this is decode-specific.** Applying this page's ratio to a prefill request, where the roofline regime is entirely different, produces a number that doesn't mean what it means here.

---

[← Back to Lecture 11 — GQA, MQA, MLA: Cheaper Attention Heads](../lectures/11-gqa-mqa-mla.md) · [Course Home](../index.md)
