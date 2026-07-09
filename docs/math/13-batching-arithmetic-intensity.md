---
layout: default
title: "Math Deep Dive — Batching, Arithmetic Intensity, and the Cost of Waiting for the Slowest"
---

# Math Deep Dive — Batching, Arithmetic Intensity, and the Cost of Waiting for the Slowest

> **Supports:** [Lecture 13 — Continuous Batching](../lectures/13-continuous-batching.md). Read the lecture first; this page assumes its static-vs-continuous simulation and its chairlift mental model.

## The Idea in One Picture

Batching helps for two completely independent reasons. Lecture 13's simulation proved the scheduling reason: idle slots are wasted capacity. This page proves the second, quieter reason — batching multiple requests' decode steps together means one weight-read now produces several tokens instead of one, which is exactly the "more FLOPs per byte moved" lever Lecture 04 built the whole roofline model around. It also explains, with real numbers, why static batching's relative waste gets *worse*, not better, as you make the batch bigger — the opposite of what naive intuition suggests.

## Notation

| Symbol | Meaning | Value used |
| --- | --- | --- |
| \\(B\\) | batch size — number of requests sharing one decode step | varies |
| \\(d\\) | a linear layer's dimension (square, for simplicity) | 4096 |
| \\(s\\) | bytes per element | 2 (bf16) |
| \\(\ell\_1, \ldots, \ell\_B\\) | i.i.d. response lengths of \\(B\\) requests in one static-batching group | drawn from Lecture 13's traffic mix |

## Derivation

### Part 1 — One weight-read, B tokens

A decode step's linear layer does \\(2d^2\\) FLOPs per token (Lecture 04's convention: 2 FLOPs per multiply-accumulate, \\(d^2\\) weights). Batch \\(B\\) requests' decode steps into one GEMM call, and the *same* weight matrix — read once from HBM — produces all \\(B\\) tokens' outputs:

\\[
\text{FLOPs}(B) = 2Bd^2, \qquad \text{bytes moved} = sd^2 \quad (\text{weight, read once, shared across the batch})
\\]

\\[
\text{AI}(B) = \frac{2Bd^2}{sd^2} = \frac{2B}{s}
\\]

\\(d\\) cancels completely, the same way it did in Lecture 04's original derivation — arithmetic intensity depends only on batch size and precision, not on model width.

### Part 2 — Where this lands on our own hardware

At \\(B=1\\), \\(s=2\\): \\(\text{AI}=1\\) — exactly Lecture 04 and Lecture 05's already-published "decode is ≈1 FLOP/byte" number, now recovered as the special case \\(B=1\\) of a more general formula.

| \\(B\\) | AI (FLOP/byte) | % of L40S ridge point (419) |
| --- | --- | --- |
| 1 | 1 | 0.2% |
| 8 | 8 | 1.9% |
| 32 | 32 | 7.6% |
| 128 | 128 | 30.5% |
| **419** | **419** | **100%** |

Solve \\(2B/s = \text{ridge}\\) for \\(B\\): at \\(s=2\\), \\(B = \text{ridge}\\) exactly — our L40S's decode arithmetic intensity crosses its own ridge point at a batch size of **419**. Below that batch size, decode is still memory-bound (just less severely, the larger \\(B\\) gets); above it, batched decode genuinely becomes compute-bound. This is a real, checkable target number, not a metaphor — Lecture 13's Exercise 4 asks you to confirm it.

### Part 3 — Why static batching's waste gets worse with bigger batches

A static batch's wall time is \\(\max(\ell\_1,\ldots,\ell\_B)\\) — the maximum of \\(B\\) random draws. A basic, general fact about maxima: **the expected maximum of \\(B\\) i.i.d. samples grows with \\(B\\), and grows faster than the mean does.** Intuitively, every additional sample is one more chance to draw an unusually large value, and nothing caps how large that value can get except the underlying distribution's own tail.

Lecture 13's own traffic mix (mean length ≈139 tokens) makes this concrete — computed directly by repeated sampling, not asserted:

| \\(B\\) | \\(E[\max]\\) | Ratio to mean |
| --- | --- | --- |
| 1 | 139.4 | 1.00× |
| 2 | 210.8 | 1.52× |
| 8 | 426.0 | 3.06× |
| 16 | 556.0 | 4.00× |
| 32 | 694.4 | 4.99× |

A static batch's *per-slot* wall time (its wasted-idle share) is roughly \\(E[\max] - E[\ell]\\), and that gap grows faster than \\(B\\) itself — bigger static batches don't just fail to fix the idle-slot problem, they make each group's relative waste worse. This is the precise, quantified version of Lecture 13's Exercise 2 observation.

## Worked Example

Using Part 2's formula directly: at what batch size does decode cross from "under 10% of peak compute" to "over 50%"? Solve \\(2B/s = 0.10\times419\\) and \\(2B/s=0.50\times419\\) at \\(s=2\\):

\\[
B\_{10\%} = 0.10 \times 419 = 41.9 \approx 42, \qquad B\_{50\%} = 0.50\times419 = 209.5 \approx 210
\\]

A batch of 42 concurrent decode steps already reaches 10% of the L40S's peak compute throughput — a real, achievable batch size for a moderately busy server — while still leaving 90% of the roofline's compute ceiling unused. This is exactly why continuous batching's *scheduling* win (Lecture 13's main result) and this page's *arithmetic-intensity* win are both worth having: neither alone gets decode anywhere near compute-bound at realistic batch sizes.

## Where the Assumptions Break

**Part 1 only covers the linear layers.** The attention operation itself has its own arithmetic-intensity story, already derived in Math Deep Dive 11 (scales with KV group size, not batch size, for a *single* request) — a fully batched decode step's overall AI blends both effects, weighted by how much time each part takes, not simply Part 1's formula alone.

**Part 3's "expected maximum" numbers are empirical for this course's specific traffic mix**, not a closed-form distributional result — the exact growth rate of \\(E[\max]\\) with \\(B\\) depends on the tail behavior of the underlying length distribution (heavier-tailed traffic makes this effect even more pronounced; a distribution with a hard upper cap makes it less so). The qualitative conclusion — bigger static batches waste relatively more, not less — holds broadly; the exact numbers in the table are specific to Lecture 13's own simulated mix.

**Real batches aren't actually i.i.d.** Production traffic has correlated bursts (many similar requests arriving together, e.g. a batch job), which changes the maximum's behavior in ways this idealized model doesn't capture.

## Common Mistakes

- **Assuming bigger static batches are strictly better because nominal throughput capacity is higher.** Part 3 shows the opposite for *relative* waste — nominal capacity and actual utilization are different numbers, and static batching's gap between them widens with \\(B\\).
- **Treating the roofline crossing point (\\(B\approx419\\)) as a target to hit.** It's a reference point for *this specific GPU and precision*, not a universal number — a different card's ridge point, or a lower/higher precision, moves it.
- **Forgetting Part 1 and Lecture 13's scheduling result are additive, not competing explanations.** Continuous batching helps because it keeps slots full (a scheduling fact) *and* because a fuller batch has higher arithmetic intensity (a roofline fact) — both are true at once, for the same underlying reason: don't waste the weight-read you already paid for.

---

[← Back to Lecture 13 — Continuous Batching](../lectures/13-continuous-batching.md) · [Course Home](../index.md)
