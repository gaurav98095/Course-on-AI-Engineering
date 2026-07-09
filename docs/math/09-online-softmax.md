---
layout: default
title: "Math Deep Dive — Online Softmax and the FlashAttention Tiling Trick"
---

# Math Deep Dive — Online Softmax and the FlashAttention Tiling Trick

> **Supports:** [Lecture 09 — FlashAttention](../lectures/09-flashattention.md). Read the lecture first; this page assumes its naive-vs-flash comparison and its chef/counter mental model.

## The Idea in One Picture

Standard softmax looks like it needs to see an entire row of scores before it can output a single number — the denominator sums over every score, and numerical stability requires subtracting the row's maximum first. That "see everything first" requirement is exactly why naive attention materializes the whole row in HBM. This page derives the trick that breaks the requirement: process the row in small pieces, keep a *running* max and a *running*, correctly-rescaled sum, and combine pieces into an answer that is — not approximately, but exactly — the same number standard softmax would have produced.

## Notation

| Symbol | Meaning |
| --- | --- |
| \\(x\_1, \ldots, x\_N\\) | the row of attention scores for one query |
| \\(m\\) | the row's maximum, \\(\max\_i x\_i\\) |
| \\(l\\) | the softmax denominator, \\(\sum\_i e^{x\_i - m}\\) |
| \\(v\_1, \ldots, v\_N\\) | the value vectors the scores weight |
| \\(O\\) | the (unnormalized) weighted sum \\(\sum\_i e^{x\_i-m} v\_i\\); the final output is \\(O/l\\) |
| \\(m\_1, l\_1, O\_1\\) and \\(m\_2, l\_2, O\_2\\) | the same three quantities, computed on two separate blocks of the row |

## Derivation

### Part 1 — Why the max subtraction exists at all

Softmax of a row is \\(\text{softmax}(x)\_i = e^{x\_i} / \sum\_j e^{x\_j}\\). Computed literally, this overflows fast — Lecture 07 already showed \\(e^{70000}\\) overflows even fp32's range. The standard fix subtracts the row's max first:

\\[
\text{softmax}(x)\_i = \frac{e^{x\_i-m}}{\sum\_j e^{x\_j-m}}, \qquad m = \max\_j x\_j
\\]

This is mathematically identical to the naive formula (the \\(e^{-m}\\) factor cancels between numerator and denominator) and numerically safe, because every exponent is now \\(\le 0\\). The catch: you need \\(m\\), the max of the *whole* row, before you can safely exponentiate even the first score.

### Part 2 — Two blocks, two sets of local statistics

Split the row into two blocks (in FlashAttention, a block is however many keys fit in SRAM at once). Compute each block's own local max, local sum, and local weighted-value-sum, entirely independently:

\\[
m\_1 = \max\_{i \in \text{block 1}} x\_i, \qquad l\_1 = \sum\_{i \in \text{block 1}} e^{x\_i - m\_1}, \qquad O\_1 = \sum\_{i \in \text{block 1}} e^{x\_i - m\_1} v\_i
\\]

— and the same for block 2. Each block only ever needs to see *its own* scores to compute these. Nothing here is correct yet as a piece of the *global* softmax — block 1 has no idea block 2 might contain a bigger score. That's what the merge step fixes.

### Part 3 — Merging two blocks exactly

The true global sum is just the sum of both blocks' contributions, re-expressed under one shared max \\(m = \max(m\_1, m\_2)\\). Each block's locally-computed sum used the *wrong* max (its own, not the global one) — correct it by multiplying by \\(e^{m\_i - m}\\), which is just re-scaling \\(e^{x - m\_i}\\) into \\(e^{x-m}\\):

\\[
l = l\_1 e^{m\_1-m} + l\_2 e^{m\_2-m}, \qquad O = O\_1 e^{m\_1-m} + O\_2 e^{m\_2-m}
\\]

The final answer is \\(O/l\\) — exactly what a single pass over all \\(N\\) scores with the true global max would have produced. No approximation happened; every rescaling factor is exact arithmetic. This is the entire trick: **you don't need the global max in advance — you can discover it as you go, and cheaply repair everything you already computed.**

### Part 4 — Streaming over many blocks

Real attention rows have far more than two blocks. The merge rule in Part 3 is associative — apply it block by block, carrying forward a running \\((m, l, O)\\), and the result after the last block is the same exact global answer:

```text
m, l, O = -inf, 0, 0
for block in blocks:
    m_new = max(m, block.local_max)
    l = l * exp(m - m_new) + block.local_l * exp(block.local_max - m_new)
    O = O * exp(m - m_new) + block.local_O * exp(block.local_max - m_new)
    m = m_new
output = O / l
```

This loop is the actual inner loop of a FlashAttention kernel: at every step, only one block's worth of scores exists anywhere — computed fresh in SRAM, folded into the running statistics, and discarded. The full \\(N\times N\\) matrix Lecture 09's naive path writes to HBM never exists all at once, anywhere.

## Worked Example

Four scores, split into two blocks of two — small enough to check by hand, verified against a direct full-row computation:

\\(x = (1.0,\ 3.0,\ 0.0,\ 2.0)\\), with scalar values \\(v = (2.0,\ -1.0,\ 4.0,\ 0.5)\\) for each score, respectively.

**Ground truth — standard softmax over all four at once:** \\(m=3.0\\); \\(e^{x-m} = (0.1353,\ 1.0,\ 0.0498,\ 0.3679)\\); \\(l = 1.553\\); output \\(= \sum e^{x\_i-m}v\_i / l = -0.223\\).

**Block 1** (\\(x=1.0, 3.0\\)): \\(m\_1=3.0\\), \\(l\_1 = e^{-2}+e^{0} = 1.135\\), \\(O\_1 = e^{-2}(2.0) + e^{0}(-1.0) = -0.729\\).

**Block 2** (\\(x=0.0, 2.0\\)): \\(m\_2=2.0\\), \\(l\_2 = e^{-2}+e^{0} = 1.135\\), \\(O\_2 = e^{-2}(4.0) + e^{0}(0.5) = 1.041\\).

**Merge:** \\(m=\max(3.0, 2.0)=3.0\\).

\\[
l = 1.135\,e^{3-3} + 1.135\,e^{2-3} = 1.135 + 0.417 = 1.553
\\]
\\[
O = -0.729\,e^{3-3} + 1.041\,e^{2-3} = -0.729 + 0.383 = -0.346
\\]

Final output: \\(O/l = -0.346 / 1.553 = -0.223\\) — matching the direct computation to three decimal places, exactly as Part 3 promised. *(These numbers are computed directly, not typed by hand — five lines of numpy reproduce them.)*

## Where the Assumptions Break

**Floating-point addition isn't associative.** The merge in Part 3 gets the *mathematically* exact answer, but summing the same numbers in a different order (all-at-once vs. block-by-block) can still land a few bits apart in the last decimal place, purely from rounding. Lecture 09's `allclose`-not-`==` sanity check exists for exactly this reason — the algorithm is exact, the floating-point *implementation* of any sum is only ever exact up to rounding, block-wise or not.

**The backward pass recomputes rather than stores.** FlashAttention's training-time trick — recomputing the forward pass's intermediate values during the backward pass instead of storing them — trades extra compute for less memory, and is a real cost. This course's inference-only lectures never pay it: there is no backward pass at serving time, so the memory-traffic win here comes with none of that recomputation tax.

**Block size is a real tuning knob, not a free parameter.** Bigger blocks mean fewer merge steps (less rescaling overhead) but need more SRAM per block; SRAM is typically under 256 KB per streaming multiprocessor on current GPUs. A block size that doesn't fit spills back to HBM and quietly loses the whole advantage — this is a kernel-engineering decision FlashAttention's authors tune per-GPU, not something this lecture's `torch.nn.functional.scaled_dot_product_attention` call exposes directly.

## Common Mistakes

- **Assuming online softmax is an approximation.** It isn't — Part 3's merge is exact algebra, not a truncation or a bound. Any numerical difference from naive softmax is ordinary floating-point rounding, not the algorithm being "close enough."
- **Forgetting the rescale when a new max arrives.** Skipping the \\(e^{m\_i-m}\\) correction and just adding raw block sums together silently produces a wrong answer — the single most common bug in a hand-written online-softmax implementation.
- **Applying this only to attention.** Online softmax is a general streaming algorithm for any softmax too large to hold in fast memory at once — attention is its most famous use, not its only one.

---

[← Back to Lecture 09 — FlashAttention](../lectures/09-flashattention.md) · [Course Home](../index.md)
