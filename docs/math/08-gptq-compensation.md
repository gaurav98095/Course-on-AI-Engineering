---
layout: default
title: "Math Deep Dive — The Compensation Formula Behind GPTQ"
---

# Math Deep Dive — The Compensation Formula Behind GPTQ

> **Supports:** [Lecture 08 — Quantization II: GPTQ & AWQ in Practice](../lectures/08-quantization-ii-gptq-and-awq.md). Read the lecture first; this page assumes its story and mental model.

## The Idea in One Picture

Naive quantization asks "how do I round this one weight?" — a question about a single number. GPTQ asks a completely different question: "given everything else in this layer, how do I round *all* the weights so the layer's actual output on real data changes as little as possible?" That's a question about a matrix, not a number, and this page derives exactly which matrix, and what to do with it.

## Notation

| Symbol | Meaning | Shape / value |
| --- | --- | --- |
| \\(w\\) | one row of a weight matrix (one output neuron's weights) | \\((d_{\text{in}},)\\) |
| \\(X\\) | calibration inputs flowing through this layer | \\((d_{\text{in}}, n)\\), \\(n\\) = calibration samples |
| \\(H\\) | \\(XX^\top\\) — the layer's Hessian w.r.t. this row's weights | \\((d_{\text{in}}, d_{\text{in}})\\) |
| \\(q\\) | index of the weight being quantized right now | scalar |
| \\(F\\) | indices of weights not yet quantized ("free") | subset of \\(\{1,\dots,d_{\text{in}}\}\\) |
| \\(e_q\\) | rounding error on weight \\(q\\), \\(w_q - \hat{w}_q\\) | scalar |
| \\(\delta_F\\) | compensating update applied to the free weights | \\((\lvert F\rvert,)\\) |

## Derivation

### Part 1 — The objective is about the output, not the weights

A linear layer computes \\(y = w^\top x\\) for each calibration sample \\(x\\), stacked as \\(y = w^\top X\\) across all \\(n\\) samples. Quantizing \\(w\\) to \\(\hat w\\) changes that output. The reconstruction-error objective GPTQ actually minimizes is how much:

\\[
\mathcal{L}(\hat w) = \lVert w^\top X - \hat w^\top X \rVert_2^2 = (w-\hat w)^\top \underbrace{XX^\top}_{H} (w - \hat w)
\\]

That's the whole idea in one line: **error is measured after multiplying by the data, not before.** A weight that calibration data barely touches can be rounded carelessly with almost no effect on \\(\mathcal{L}\\); a weight that gets multiplied by large, frequent activations cannot. \\(H = XX^\top\\) is exactly this sensitivity, captured once and reused for every output row, because it depends only on the *inputs* \\(X\\), never on the weights.

### Part 2 — Quantize one weight, solve for the optimal compensation

Quantize weight \\(q\\) by rounding it to the nearest representable value, \\(\hat w_q\\), leaving error \\(e_q = w_q - \hat w_q\\). This is now fixed — it already happened. The question is: how should the *still-free* weights \\(F\\) move to make up for it, minimizing the damage to \\(\mathcal{L}\\)?

This is a constrained minimization: minimize the quadratic form \\(\mathcal{L}\\) subject to weight \\(q\\)'s change being exactly \\(-e_q\\) (since it just got rounded by that much) and every weight in \\(F\\) free to move. Solving this with Lagrange multipliers (the standard tool for "minimize subject to one coordinate being pinned") yields a clean closed form — this is the same derivation underlying classical *Optimal Brain Surgeon* pruning, adapted here to quantization instead of removal:

\\[
\boxed{\delta_F = -\,e_q\,\frac{[H^{-1}]_{F,q}}{[H^{-1}]_{q,q}}}
\\]

Read it as: take the error we just introduced, and distribute a share of it to every free weight, proportional to how strongly that weight is coupled to weight \\(q\\) *through the inverse Hessian* — which encodes both direct and indirect (via other correlated features) sensitivity, not just raw correlation.

### Part 3 — Why GPTQ isn't quite the optimal (OBQ) algorithm

The fully optimal version of this idea — Optimal Brain Quantization (OBQ) — picks, at every step, *whichever remaining weight is cheapest to quantize next*, re-deriving \\(H^{-1}\\) after each one. That is provably optimal, and far too slow for an 8-billion-parameter model: re-inverting a shrinking Hessian thousands of times per layer doesn't finish in a human lifetime at that scale.

GPTQ's practical shortcut: quantize weights in a **fixed, arbitrary order** (left to right) instead of the optimal order, and maintain \\(H^{-1}\\) incrementally via a Cholesky decomposition computed once per layer rather than re-inverting from scratch at every step. The compensation formula in Part 2 is unchanged; only *which weight gets quantized when* is simplified. Frantar et al.'s empirical result — the actual justification for this shortcut — is that final quantization quality barely suffers from fixing the order, while runtime drops from impractical to "several minutes," exactly what Lecture 08's `quantize_gptq_awq.py` run demonstrated.

## Worked Example

A tiny 2-weight row, small enough to check by hand. Calibration data: three samples through a 2-input layer,

\\[
X = \begin{pmatrix} 1 & 0 & 1 \\\\ 0 & 2 & 1 \end{pmatrix}
\\]

(columns are samples; row 1 is feature 1's value across samples, row 2 is feature 2's).

**Build H:**

\\[
H = XX^\top = \begin{pmatrix} 2 & 1 \\\\ 1 & 5 \end{pmatrix}
\\]

**Invert it.** For a 2×2 matrix \\(\begin{pmatrix} a & b \\\\ c & d \end{pmatrix}\\), the inverse is \\(\frac{1}{ad-bc}\begin{pmatrix} d & -b \\\\ -c & a \end{pmatrix}\\); here the determinant is \\(2\cdot5-1\cdot1=9\\):

\\[
H^{-1} = \frac{1}{9}\begin{pmatrix} 5 & -1 \\\\ -1 & 2 \end{pmatrix}
\\]

**Quantize the row** \\(w = (1.3,\ -0.7)\\) onto a toy 0.5-spaced grid \\(\{\dots,-1.0,-0.5,0,0.5,1.0,1.5,\dots\}\\), weight 1 first. Nearest grid point to \\(1.3\\) is \\(1.5\\), so \\(\hat w_1 = 1.5\\) and \\(e_1 = 1.3 - 1.5 = -0.2\\).

**Compensate weight 2** using Part 2's formula, with \\(q=1\\), \\(F=\{2\}\\):

\\[
\delta_2 = -e_1 \cdot \frac{[H^{-1}]_{2,1}}{[H^{-1}]_{1,1}} = -(-0.2)\cdot\frac{-1/9}{5/9} = -(-0.2)\cdot\left(-\frac{1}{5}\right) = -0.04
\\]

The free weight moves from \\(-0.7\\) to \\(-0.7 + (-0.04) = -0.74\\) *before* it gets quantized. In this particular toy example both \\(-0.7\\) and \\(-0.74\\) happen to round to the same grid point (\\(-0.5\\)) — the compensation was real but too small to flip the outcome here. Notice the compensation moved weight 2 *further* from zero, not toward it: the direction follows the sign structure of \\(H^{-1}\\), not a simple "push opposite the error" rule. With a real row of 4,096 weights instead of 2, hundreds of small compensations like this one accumulate — and routinely *do* flip which grid point a weight lands on, in a direction naive per-weight rounding could never discover, because naive rounding never looks at \\(H\\) at all.

## Where the Assumptions Break

**The Hessian assumes calibration data represents real usage.** \\(H = XX^\top\\) is only a faithful sensitivity measure if the calibration inputs \\(X\\) resemble what the model will actually see in production — precisely why Lecture 08 calibrated on our own manual chunks rather than generic text.

**Fixed quantization order is a real approximation, not free.** GPTQ's speed comes directly from giving up OBQ's optimal ordering; on some layers, some models, this measurably costs a small amount of quality relative to true OBQ. In practice this cost is judged worth the (roughly) thousand-fold runtime improvement — but it *is* a trade, not a free simplification.

**Numerical stability requires care at scale.** Directly forming and inverting \\(H\\) for a 4096-dimensional row is numerically fragile; real implementations use a Cholesky decomposition and small damping terms added to \\(H\\)'s diagonal to keep the inversion well-behaved. The clean formula in Part 2 is the mathematical idea; production GPTQ code has extra machinery purely to compute it reliably in floating point.

## Common Mistakes

- **Confusing GPTQ's "Hessian" with a training-loss Hessian.** There's no loss function or gradient descent here — \\(H = XX^\top\\) comes purely from calibration *activations*, a much cheaper object than a true training Hessian, and it's specific to reconstructing this one layer's output.
- **Assuming the compensation always reduces error at every single weight.** It minimizes the *total* quadratic form across all free weights — some individual weights can end up slightly worse off if that's what's globally optimal for the layer's output.
- **Treating GPTQ as data-free.** Every step of this derivation depends on \\(X\\) — quantization quality is only as good as how representative the calibration set is, exactly as Lecture 08's "Where It Breaks" warns.

---

[← Back to Lecture 08 — Quantization II: GPTQ & AWQ in Practice](../lectures/08-quantization-ii-gptq-and-awq.md) · [Course Home](../index.md)
