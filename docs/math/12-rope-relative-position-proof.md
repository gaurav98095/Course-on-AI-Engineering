---
layout: default
title: "Math Deep Dive — RoPE as Rotation: The Relative-Position Proof"
---

# Math Deep Dive — RoPE as Rotation: The Relative-Position Proof

> **Supports:** [Lecture 12 — RoPE, ALiBi, YaRN: Positional Encodings](../lectures/12-rope-alibi-yarn.md). Read the lecture first; this page assumes its rotation formula and its numerical verification.

## The Idea in One Picture

Lecture 12 verified, numerically, that a RoPE-rotated attention score depends only on relative position. This page proves it — not by checking many examples, but by an exact algebraic identity. The whole proof rests on one fact: a 2D rotation is multiplication by a complex number of unit length, and multiplying two rotated complex numbers together makes their *individual* angles disappear, leaving only their *difference*.

## Notation

| Symbol | Meaning |
| --- | --- |
| \\(q, k\\) | one 2D pair of a query/key vector, written as a complex number \\(q = q\_1 + i q\_2\\) |
| \\(\theta\\) | that pair's rotation frequency (one of RoPE's 64 per our model's `head_dim=128`) |
| \\(m, n\\) | the query's and key's positions in the sequence |
| \\(R(p)\\) | rotation by angle \\(p\theta\\) — multiplication by \\(e^{ip\theta}\\) in the complex representation |
| \\(\bar k\\) | complex conjugate of \\(k\\) |

## Derivation

### Part 1 — A 2D rotation is complex multiplication

Write a 2D vector \\((x\_1, x\_2)\\) as the complex number \\(x = x\_1 + ix\_2\\). Multiplying by \\(e^{i\theta} = \cos\theta + i\sin\theta\\):

\\[
x \cdot e^{i\theta} = (x\_1\cos\theta - x\_2\sin\theta) + i(x\_1\sin\theta + x\_2\cos\theta)
\\]

Compare this to Lecture 12's rotation code line by line — `out[0] = x1*cos - x2*sin` is exactly the real part above, `out[1] = x1*sin + x2*cos` is exactly the imaginary part. RoPE's rotation *is* complex multiplication by \\(e^{i\theta}\\); the code just writes out the real and imaginary parts by hand.

### Part 2 — A real 2D dot product is the real part of a complex product

For two complex numbers \\(a = a\_1+ia\_2\\), \\(b=b\_1+ib\_2\\):

\\[
\text{Re}[a\bar b] = \text{Re}[(a\_1+ia\_2)(b\_1-ib\_2)] = a\_1 b\_1 + a\_2 b\_2
\\]

— exactly the ordinary real 2D dot product. So computing an attention score from two rotated 2D pairs is the same operation as taking the real part of a complex product.

### Part 3 — Rotate both, multiply, and watch \\(m\\) and \\(n\\) vanish

The query's pair, rotated to position \\(m\\), is \\(q\cdot e^{im\theta}\\). The key's pair, rotated to position \\(n\\), is \\(k \cdot e^{in\theta}\\). Their dot product, by Part 2:

\\[
\langle R(m)q,\ R(n)k\rangle = \text{Re}\Big[\big(q\,e^{im\theta}\big)\overline{\big(k\,e^{in\theta}\big)}\Big] = \text{Re}\Big[q\bar k \cdot e^{im\theta}e^{-in\theta}\Big] = \text{Re}\Big[q\bar k\cdot e^{i(m-n)\theta}\Big]
\\]

\\(m\\) and \\(n\\) enter *only* through the combination \\(e^{im\theta}e^{-in\theta} = e^{i(m-n)\theta}\\) — the individual positions are gone the moment the two exponentials combine. Whatever \\(q\\), \\(k\\), or \\(\theta\\) are, the result is a function of \\((m-n)\\) alone. This is not an approximation or a special case — it's true for every one of the 64 rotating pairs independently, and the full multi-dimensional attention score is just the sum of all 64 pairs' contributions, each one already relative-position-only.

## Worked Example

One pair, clean numbers, checked two ways. \\(\theta=0.5\\) rad/position, \\(q=(0.6, 0.8)\\), \\(k=(1.0, 0.0)\\) (both unit vectors), at \\(m=10\\), \\(n=3\\) (\\(m-n=7\\)):

**Direct rotation:** rotate \\(q\\) by \\(10\times0.5=5.0\\) rad, rotate \\(k\\) by \\(3\times0.5=1.5\\) rad, take their dot product: **\\(-0.281247\\)**.

**Complex formula:** \\(q\bar k = 0.6+0.8i\\) (magnitude 1, angle \\(0.9273\\) rad); multiply by \\(e^{i\theta(m-n)} = e^{i \cdot 3.5}\\); take the real part: **\\(-0.281247\\)** — identical.

**Same relative distance, different absolute positions** (\\(m=110, n=103\\), still \\(m-n=7\\)): direct rotation gives **\\(-0.281247\\)** again, exactly. *(All three numbers computed directly, not by hand — five lines of Python reproduce them.)*

## Where the Assumptions Break

**This proves the score depends only on \\(m-n\\) — it says nothing about how large that dependency is at long range.** Lecture 12's Part 3 showed a separate, equally real fact: a slow-rotating pair's angle range over a long context can be a small fraction of a full rotation, meaning many distinct large relative distances map to very similar angles. The exact-invariance proof on this page and the extrapolation-risk argument in the lecture are both true and don't contradict each other — one is about *what* the score depends on, the other is about *how much room* a given frequency has to represent it.

**Real RoPE implementations rotate every pair with the query's absolute position and the key's absolute position separately, then take the full dot product across all dimensions at once** — this page's per-pair algebra is exact, but a real kernel never explicitly computes \\(m-n\\); it emerges from the sum, not from an explicit subtraction anywhere in the code.

**This says nothing about what the *network* learned to do with that relative angle.** The proof is purely about the geometry RoPE guarantees; whether attending more or less strongly at a given relative distance is *useful* is learned during training, same as any other weight.

## Common Mistakes

- **Thinking RoPE encodes absolute position somewhere.** It doesn't, by Part 3's exact result — no term anywhere in the final score depends on \\(m\\) or \\(n\\) individually, only their difference.
- **Assuming the proof requires small angles or approximations.** It's an exact trigonometric identity (\\(e^{ia}e^{-ib}=e^{i(a-b)}\\)), valid for any \\(\theta\\), any \\(m\\), any \\(n\\) — nothing here is a small-angle or first-order approximation.
- **Conflating "exact relative-position invariance" with "safe to extrapolate to any length."** They're different claims — see Where the Assumptions Break above and Lecture 12's Part 3.

---

[← Back to Lecture 12 — RoPE, ALiBi, YaRN: Positional Encodings](../lectures/12-rope-alibi-yarn.md) · [Course Home](../index.md)
