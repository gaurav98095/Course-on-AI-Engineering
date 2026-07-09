---
layout: default
title: "Math Deep Dive — Why Quantized Speed Isn't Free the Way Quantized Memory Is"
---

# Math Deep Dive — Why Quantized Speed Isn't Free the Way Quantized Memory Is

> **Supports:** [Lecture 07 — Quantization I: Number Formats](../lectures/07-quantization-i-number-formats.md). Read the lecture first; this page assumes its story and mental model.

## The Idea in One Picture

Lecture 04's roofline model says decode time is proportional to bytes moved. Quantization moves fewer weight bytes — so far, so good. But "bytes moved" was never *only* the weight bytes: somewhere, a low-precision number has to become a real number the GEMM can multiply with. If that conversion itself touches memory, it adds bytes back in — sometimes more than quantization saved in the first place. This page makes that trade-off exact.

## Notation

| Symbol | Meaning | Lecture 07's values |
| --- | --- | --- |
| \\(d\\) | model hidden dimension | 4096 (course model) |
| \\(t\\) | tokens in this forward pass | 1 for decode |
| \\(s_{\text{high}}\\) | bytes/element of the compute format | 2 (bf16) |
| \\(s_{\text{low}}\\) | bytes/element of the stored (quantized) format | 1 (int8), 0.5 (int4) |
| \\(c\\) | dequantization overhead, in units of "extra full-precision tensor passes" | 0 = perfectly fused, 1 = one full write+read round trip |
| \\(B\\) | total bytes moved for one weight tensor's use | depends on \\(c\\) |

## Derivation

### Part 1 — Extending Lecture 04's byte count with a dequant term

Lecture 04's Part 2 derived bytes moved for a weight-dominant linear layer as \\(B_{\text{bf16}} \approx s_{\text{high}}\, d^2\\) — just the cost of reading the weight once. That formula silently assumed the weight arrives ready to multiply. A quantized weight doesn't: it must be unpacked back toward \\(s_{\text{high}}\\) precision before (or during) the matmul.

Model the unpacking's cost honestly, as extra full-tensor passes at the *higher* precision:

\\[
B\_{\text{quant}} = \underbrace{s\_{\text{low}}\, d^2}\_{\text{read the packed weight}} \;+\; \underbrace{c \cdot s\_{\text{high}}\, d^2}\_{\text{dequantize: }c\text{ extra passes}}
\\]

\\(c=0\\) means the GEMM kernel dequantizes on the fly, in registers or shared memory, never touching HBM again — the ideal case, and the entire point of a *purpose-built* quantized kernel (Lecture 08's subject). \\(c=1\\) means the dequantized weight gets written to HBM once and read back once more before the matmul can use it — a very plausible cost for a naive, unfused implementation.

### Part 2 — When does quantization actually win?

Quantization helps only when \\(B_{\text{quant}} < B_{\text{bf16}}\\):

\\[
s_{\text{low}}\, d^2 + c\, s_{\text{high}}\, d^2 \;<\; s_{\text{high}}\, d^2
\\]

Divide through by \\(d^2\\) and solve for \\(c\\):

\\[
\boxed{c \;<\; 1 - \frac{s_{\text{low}}}{s_{\text{high}}}}
\\]

This single inequality is the whole lecture's empirical mystery, resolved. The right-hand side is the **overhead budget** a format can tolerate before it stops paying off — and that budget is *not the same* for every quantized format.

### Part 3 — Why int4 tolerates overhead that int8 can't

Evaluate the budget for each format in the lecture:

\\[
\text{int8:}\quad 1 - \frac{1}{2} = 0.5
\qquad\qquad
\text{int4:}\quad 1 - \frac{0.5}{2} = 0.75
\\]

int8 has only a 0.5-pass budget: **any** dequantization overhead at or above a single half-tensor-equivalent pass erases the win entirely — and a naive write-then-read round trip (\\(c=1\\)) blows straight through that budget, landing exactly where Lecture 07's measurement did: no speedup, or a slight slowdown. int4 has a 0.75-pass budget — half again as much room — because it started from a smaller \\(s_{\text{low}}\\) and therefore has more savings left to spend on overhead before crossing zero.

The general pattern, provable the same way for any pair of formats: **the more aggressive the quantization, the more dequantization overhead it can absorb before losing to the unquantized baseline.** This is precisely why int4 shows a real (if modest) speedup while int8 often shows none, using the exact same class of naive kernel.

## Worked Example

Course model, \\(d=4096\\), bf16 baseline (\\(s_{\text{high}}=2\\)).

**int8, naive kernel (\\(c=1\\)):** \\(B_{\text{quant}} = 1\cdot d^2 + 1\cdot2\cdot d^2 = 3d^2\\), versus \\(B_{\text{bf16}}=2d^2\\). That's **1.5× more bytes than doing nothing at all** — a real slowdown, matching Lecture 07's measured int8 result almost exactly.

**int4, naive kernel (\\(c=1\\)):** \\(B_{\text{quant}} = 0.5d^2 + 1\cdot2\cdot d^2 = 2.5d^2\\), versus \\(B_{\text{bf16}}=2d^2\\) — still 25% *worse* than bf16 at this same overhead level. For int4 to show the **modest net win** Lecture 07 measured, its real-world \\(c\\) must be smaller than 1 — consistent with `bitsandbytes`' NF4 path being a more optimized, more fused implementation than its older int8 path. Solve for the \\(c\\) that would make int4 exactly break even with bf16: \\(0.5 + 2c = 2 \Rightarrow c = 0.75\\); anything below that shows a real speedup, which is exactly the 0.75 budget derived in Part 3.

**A perfectly fused kernel (\\(c=0\\), either format):** \\(B_{\text{quant}} = s_{\text{low}}\,d^2\\) exactly — the full, uncompromised roofline prediction from Lecture 07's "Math, One Level Deeper" section. This is the target Lecture 08's GPTQ/AWQ kernels are built to hit.

## Where the Assumptions Break

**\\(c\\) isn't a fixed hardware constant — it's a property of the *kernel implementation*.** The same GPU, the same format, can show wildly different \\(c\\) depending on whether the library fuses dequantization into the matmul or not. This is precisely why "is quantization faster" has no single answer independent of *which* quantized inference engine you're using.

**Batching changes the picture.** Recall Lecture 04's Part 4: batching decode across \\(B\\) requests multiplies \\(t\\), pushing arithmetic intensity up regardless of \\(s\\). At large enough batch, the workload can cross into compute-bound territory where this entire memory-bound analysis stops applying — the dequant overhead then competes against compute time, not bandwidth, and the trade-offs shift again.

**Real dequantization sometimes reuses cache.** L2 cache (Lecture 04) can make a "second pass" over recently-touched data much cheaper than a full HBM round trip — so \\(c\\) in practice often sits between the idealized 0 and the pessimistic 1 assumed here, and depends on tensor size relative to cache size.

## Common Mistakes

- **Assuming quantized memory savings imply quantized speed savings.** They're governed by different math entirely — Part 2's inequality is not automatically satisfied just because \\(s_{\text{low}} < s_{\text{high}}\\).
- **Blaming the *format* when the *kernel* is the problem.** int8 "not being faster" is a statement about a specific unfused implementation's \\(c\\), not a law about 8-bit integers.
- **Ignoring that more aggressive quantization has more error-tolerance for overhead too.** It's not just "int4 saves more bytes" — Part 3 shows it also *tolerates more inefficiency* before losing, which is a second, independent reason it wins more often in practice.

---

[← Back to Lecture 07 — Quantization I: Number Formats](../lectures/07-quantization-i-number-formats.md) · [Course Home](../index.md)
