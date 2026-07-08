---
layout: default
title: "Math Deep Dive — Deriving the Roofline Model"
---

# Math Deep Dive — Deriving the Roofline Model

> **Supports:** [Lecture 04 — The GPU: Architecture, HBM, and the Roofline Model](../lectures/04-the-gpu-architecture-and-roofline.md). Read the lecture first; this page assumes its story and mental model.

## The Idea in One Picture

A GPU can only do two things to finish a workload: compute, and move bytes. Each has a top speed. Whichever top speed you hit first is your ceiling. Plot both ceilings on log-log axes against "bytes moved per unit of work" and you get two straight lines that cross exactly once — the roofline.

## Notation

| Symbol | Meaning | Typical value (L40S) |
| --- | --- | --- |
| \\(F\\) | total FLOPs a workload performs | depends on the op |
| \\(B\\) | total bytes moved between HBM and the chip | depends on the op |
| \\(\text{AI}\\) | arithmetic intensity, \\(F/B\\) | 1 (decode) to 1000s (prefill) |
| \\(\pi\\) | peak compute (TFLOP/s) | ~362 (fp16 dense, datasheet) |
| \\(\beta\\) | peak bandwidth (GB/s) | 864 (datasheet) |
| \\(R\\) | ridge point, \\(\pi/\beta\\) (FLOPs/byte) | ~419 |
| \\(d\\) | model hidden dimension | read from config, e.g. 4096 |
| \\(t\\) | tokens processed in one forward pass | 1 (decode) to thousands (prefill) |

## Derivation

### Part 1 — Two ceilings, two straight lines

**The compute ceiling** is simple: a GPU cannot exceed its peak FLOP/s, no matter what:

\\[
\text{achievable TFLOP/s} \le \pi
\\]

**The bandwidth ceiling** is a little more interesting. In time \\(T\\), the chip can move at most \\(\beta T\\) bytes. If the workload has arithmetic intensity \\(\text{AI} = F/B\\), then the FLOPs it can complete using those bytes is \\(F = \text{AI} \times B \le \text{AI} \times \beta T\\). Divide by \\(T\\):

\\[
\text{achievable TFLOP/s} \le \text{AI} \times \beta
\\]

Put the two ceilings together — you can never beat *either* one:

\\[
\boxed{\text{achievable TFLOP/s} = \min(\pi,\ \text{AI} \times \beta)}
\\]

On log-log axes (log TFLOP/s vs. log AI), \\(\text{AI} \times \beta\\) is a straight line of slope 1 (the memory-bound region), and \\(\pi\\) is a flat horizontal line (the compute-bound region). They cross where \\(\text{AI} \times \beta = \pi\\), i.e. at \\(\text{AI} = \pi/\beta = R\\), the ridge point. That crossing is exactly the diagram in the lecture.

### Part 2 — Arithmetic intensity of a linear layer, in general

Take one matmul \\(Y = XW\\), with \\(X\\) of shape \\((t, d_{\text{in}})\\) and \\(W\\) of shape \\((d_{\text{in}}, d_{\text{out}})\\). Standard FLOP counting for matrix multiply (one multiply + one add per output element per reduction step):

\\[
F = 2\, t\, d_{\text{in}}\, d_{\text{out}}
\\]

Bytes moved, assuming the weight is read once from HBM per forward pass (true for decode; we address the multi-request case below) and \\(s\\) bytes per element (\\(s=2\\) for fp16):

\\[
B = \underbrace{s\, d_{\text{in}} d_{\text{out}}}_{\text{weight}} + \underbrace{s\, t\, d_{\text{in}}}_{\text{read }X} + \underbrace{s\, t\, d_{\text{out}}}_{\text{write }Y}
\\]

For a square layer (\\(d_{\text{in}} = d_{\text{out}} = d\\)) and small \\(t\\), the weight term dominates completely (\\(d^2 \gg t d\\) whenever \\(t \ll d\\) — true for \\(t=1\\) and \\(d\\) in the thousands):

\\[
B \approx s\, d^2 \qquad\Longrightarrow\qquad \text{AI} = \frac{2\,t\,d^2}{s\,d^2} = \frac{2t}{s}
\\]

**Arithmetic intensity of a linear layer scales linearly with the number of tokens processed together — and doesn't depend on \\(d\\) at all.** That is the entire mathematical content of "decode is slow, prefill is fast": \\(t=1\\) gives \\(\text{AI} = 2/s = 1\\) (fp16); \\(t=2048\\) gives \\(\text{AI} \approx 2048\\).

### Part 3 — Solving for the crossover token count

Set the layer's AI equal to the ridge point and solve for \\(t\\):

\\[
\frac{2t}{s} = R \qquad\Longrightarrow\qquad t^\* = \frac{sR}{2}
\\]

For our L40S (\\(R \approx 419\\), \\(s=2\\)): \\(t^\* \approx 419\\) tokens. Below ~419 tokens in a single forward pass, you are memory-bound; above it, compute-bound. This is precisely why our `roofline.py` sweep flips regime between \\(t=128\\) and \\(t=512\\) — exercise 3 in the lecture asks you to find this crossing on your own hardware and check it against this formula.

### Part 4 — Why more concurrent requests is the same lever as more tokens

Continuous batching (Lecture 13) stacks \\(B\\) independent requests' single tokens into one \\((B, d)\\) matrix before the matmul. The weight is still read **once**, but now \\(t = B\\) decode tokens ride along. Re-run Part 2 with \\(t = B\\):

\\[
\text{AI} = \frac{2B}{s}
\\]

Identical formula to prefill's — batching many *requests'* decode steps together is mathematically the same move as batching many *tokens* of one prompt together. This is the single equation that explains why serving many users at once doesn't just help fairness (Lecture 03); it directly increases arithmetic intensity, pushing the workload rightward toward the compute ceiling and buying real throughput. Module 2 is, in a real sense, one extended argument for maximizing \\(t\\) by every means available.

## Worked Example

Course model, \\(d = 4096\\) (read from the real config), fp16 (\\(s=2\\)), our Studio's measured ridge point \\(R \approx 397\\).

**Decode**, \\(t=1\\): \\(\text{AI} = 2(1)/2 = 1\\) FLOP/byte. Since \\(1 \ll 397\\), achievable \\(= \text{AI} \times \beta = 1 \times 780\text{GB/s} = 0.78\\) TFLOP/s — a rounding error against the card's ~310 TFLOP/s measured ceiling. This is why decode-shaped work can never get faster by adding more FLOP/s to the chip; it needs more bytes/s, or fewer bytes moved per useful FLOP (which is exactly what quantization, Lecture 07, delivers).

**Prefill**, \\(t=1800\\) (our RAG's real prompt size from Lecture 01): \\(\text{AI} = 2(1800)/2 = 1800\\) FLOPs/byte. Since \\(1800 \gg 397\\), achievable \\(\approx \pi \approx 310\\) TFLOP/s — the compute ceiling, full stop. This is why doubling HBM bandwidth would do *nothing* for our TTFT, while doubling Tensor Core throughput would cut it roughly in half.

**Crossover check**: \\(t^\* = sR/2 = 2(397)/2 = 397\\) tokens — matching the lecture's observed flip between \\(t=128\\) (still memory-bound) and \\(t=512\\) (already compute-bound).

## Where the Assumptions Break

**Perfect overlap is assumed, never guaranteed.** The `min()` formula presumes memory transfers and compute can be perfectly pipelined — the next tile's bytes streaming in while the current tile's Tensor Cores are busy. Real kernels lose some fraction to launch overhead, imperfect tiling, and synchronization; 60–90% of the theoretical roofline is a realistic ceiling for good code, not 100%.

**One bandwidth number hides a memory hierarchy.** We treated "memory" as a single HBM number, but L2 cache offers far higher bandwidth to data that's already resident. A kernel with poor cache reuse can be bandwidth-bound against *L2*, not HBM, while our HBM-only roofline says it should be comfortably compute-bound. `ncu` (Lecture 06) profiles this directly; the roofline model here is the first, coarsest cut.

**Attention isn't a plain GEMM.** Self-attention's own arithmetic intensity depends on sequence length in a more complex way (softmax, the extra \\(QK^\top\\) and \\(\text{scores}\cdot V\\) passes), and naive implementations move far more bytes than the FLOP count alone suggests — exactly the inefficiency FlashAttention (Lecture 09) is built to remove.

**FP8/INT4 change \\(s\\), not just precision.** Lower-precision weights mean fewer bytes per element — \\(s\\) shrinks, so for fixed \\(t\\) the arithmetic intensity **rises** for the same amount of useful work. This is the quantitative reason quantization helps decode specifically: it's not just "smaller model," it's "higher arithmetic intensity, same token."

## Common Mistakes

- **Quoting datasheet peaks as achieved performance.** Datasheet TFLOP/s and GB/s are upper bounds under ideal conditions; always measure your own kernel's achieved numbers, as `roofline.py` does.
- **Assuming a "big" model is automatically compute-bound.** Arithmetic intensity of decode is \\(2t/s\\) — independent of \\(d\\) entirely. A 70B model at batch 1 is exactly as memory-bound as a 1B model at batch 1.
- **Forgetting activation bytes entirely.** They're negligible for \\(t \ll d\\) (our approximation), but at large batch/long sequence they matter — don't reuse the decode-regime formula unmodified for large-batch prefill analysis without checking the full \\(B\\) expression from Part 2.
- **Treating the ridge point as universal.** It's a property of a specific GPU at a specific precision. An FP8 ridge point is a different number from an FP16 ridge point on the *same* card, because \\(\pi\\) changes with precision while \\(\beta\\) mostly doesn't.

---

[← Back to Lecture 04 — The GPU: Architecture, HBM, and the Roofline Model](../lectures/04-the-gpu-architecture-and-roofline.md) · [Course Home](../index.md)
