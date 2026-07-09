---
layout: default
title: "Math Deep Dive — Amdahl's Law and Where to Spend an Hour"
---

# Math Deep Dive — Amdahl's Law and Where to Spend an Hour

> **Supports:** [Lecture 06 — Profiling: Where the Time Actually Goes](../lectures/06-profiling-where-the-time-actually-goes.md). Read the lecture first; this page assumes its story and mental model.

## The Idea in One Picture

A request is a sum of parts. Speeding up one part by any amount — even making it instant — can only ever save the time that part used to take. If a part was small to begin with, no amount of heroics on it moves the total by much. Amdahl's Law turns that observation into an exact number, so "where should I spend the next hour" stops being a guess.

## Notation

| Symbol | Meaning | Our profile table |
| --- | --- | --- |
| \\(T\\) | total time of the original request | ~1,060 ms |
| \\(p\\) | fraction of \\(T\\) spent in the part being optimized | e.g. 0.004 for retrieval |
| \\(1-p\\) | fraction spent in everything else (untouched) | 0.996 for retrieval's case |
| \\(s\\) | speedup factor applied to that one part | \\(s \to \infty\\) = "made instant" |
| \\(\text{Speedup}\\) | ratio of old total time to new total time | what we're solving for |

## Derivation

### Part 1 — Splitting the request into two pieces

Take total time \\(T\\) and split it into the part you're about to optimize and everything else:

\\[
T = \underbrace{(1-p)T}\_{\text{unchanged}} + \underbrace{pT}\_{\text{about to speed up}}
\\]

Speed up only the second piece by a factor of \\(s\\) — it now takes \\(pT/s\\) instead of \\(pT\\). The new total time is:

\\[
T_{\text{new}} = (1-p)T + \frac{pT}{s}
\\]

### Part 2 — The speedup formula

Speedup is old time over new time. Divide through by \\(T\\) and simplify:

\\[
\text{Speedup} = \frac{T}{T_{\text{new}}} = \frac{T}{(1-p)T + \dfrac{pT}{s}} = \boxed{\dfrac{1}{(1-p) + \dfrac{p}{s}}}
\\]

Every symbol in that boxed formula is something you can read straight off a profiler table: \\(p\\) is a row's percentage of total time; \\(s\\) is however much faster you can make that row.

### Part 3 — The ceiling as \\(s \to \infty\\)

The most useful special case: what's the *best possible* outcome, if you could make one part infinitely fast? Let \\(s \to \infty\\), so \\(p/s \to 0\\):

\\[
\text{Speedup}_{\max} = \frac{1}{1-p}
\\]

This is a hard ceiling that no amount of cleverness on that one part can cross. It depends **only** on \\(p\\) — how big the part already was — never on how good your optimization is.

| \\(p\\) (fraction of total time) | Max possible speedup, however hard you try |
| --- | --- |
| 0.004 (retrieval, our profile) | 1.004× — 0.4% |
| 0.10 | 1.11× |
| 0.30 | 1.43× |
| 0.72 (matmul+linear, our profile) | 3.57× |
| 0.95 | 20× |

Read the table as a policy, not just arithmetic: a part must already be a **large fraction** of the total before it's worth heavy optimization effort, regardless of how satisfying that optimization would be to write.

### Part 4 — Generalizing to many parts

Real systems have more than two pieces. Split total time into \\(n\\) fractions \\(p_1, p_2, \ldots, p_n\\) (summing to 1), each with its own achievable speedup \\(s_i\\):

\\[
\text{Speedup} = \frac{1}{\displaystyle\sum_{i=1}^{n} \dfrac{p_i}{s_i}}
\\]

This is the same identity, just summed over every row of a profiler table at once — it's exactly the tool for deciding, from one profile, which of *several* candidate optimizations is worth doing first: rank rows by \\(p_i \times (\text{how easily you can shrink that row})\\), not by which row looks most interesting to fix.

## Worked Example

Using this lecture's profile table (retrieval 0.4%, matmul 58.1%, attention 19.4%, linear 14.2%, everything else the remainder):

**Optimizing retrieval to zero:** \\(p = 0.004\\), \\(s \to \infty\\). Speedup \\(= 1/(1-0.004) \approx 1.004\\) — a 0.4% win, however brilliant the retrieval optimization.

**Optimizing matmul by 2×** (a realistic outcome from, say, quantization in Lecture 07): \\(p = 0.581\\), \\(s = 2\\).

\\[
\text{Speedup} = \frac{1}{(1-0.581) + \dfrac{0.581}{2}} = \frac{1}{0.419 + 0.2905} = \frac{1}{0.7095} \approx 1.41\times
\\]

**Optimizing matmul all the way to compute-optimal** (Lecture 04's ceiling, roughly \\(s=3\\) given our measured 18.4% compute-throughput headroom to 100%): \\(p=0.581\\), \\(s=3\\).

\\[
\text{Speedup} = \frac{1}{0.419 + 0.581/3} = \frac{1}{0.419+0.194}=\frac{1}{0.613}\approx 1.63\times
\\]

Compare the three: a *heroic, perfect* fix to retrieval (0.4% of the request) caps out at 1.004×. A *modest* 2× win on matmul — the kind quantization routinely delivers — already beats it more than thirty-fold. This is the exact arithmetic behind "profile before you optimize," not just good advice.

## Where the Assumptions Break

**\\(p\\) isn't fixed once you start optimizing other things.** Speed up decode dramatically (Module 2 does exactly this) and prefill's *share* of total time rises even though prefill itself didn't change — Amdahl's Law describes one optimization holding everything else constant, not a whole roadmap. Re-profile after each change; a part's \\(p\\) shifts under you.

**A "small" \\(p\\) can still gate everything else.** If retrieval were on the *critical path* in a way that blocked generation from starting (it isn't, in our pipeline — they're sequential but retrieval is simply fast), a small \\(p\\) with a hidden dependency can matter far more than its raw percentage suggests. Amdahl's Law assumes the parts are cleanly separable in time, which our profile table happens to satisfy — always sanity-check that assumption before applying the formula blindly.

**This is a ceiling, not a promise.** \\(s \to \infty\\) is never actually achievable — real optimizations have their own floors (a memory-bound kernel can't beat its bandwidth ceiling, per Lecture 04's roofline). Treat \\(\text{Speedup}_{\max} = 1/(1-p)\\) as "the most this could ever be worth," a filter for ruling *out* low-value work, not a target to hit.

## Common Mistakes

- **Optimizing the most interesting row instead of the biggest one.** A clever fix to a 2% row is still capped near 1.02× — check \\(p\\) before committing engineering time, regardless of how elegant the fix would be.
- **Forgetting \\(s\\) has its own ceiling.** A "2× speedup" claim needs its own justification (a roofline bound, a benchmark) — Amdahl's Law tells you what a given \\(s\\) is *worth*, not how to achieve it.
- **Applying single-part Amdahl's Law after multiple rounds of optimization.** Once you've already sped up matmul, its \\(p\\) in the *new* profile is smaller than in the *old* one — always re-measure \\(p\\) from the current system, not the original.
- **Treating the formula as approximate.** It's an exact algebraic identity given the (1-p)/p split — any discrepancy against a real measurement means the split itself was wrong (hidden overlap, a part not actually independent), not that the formula is "just a rule of thumb."

---

[← Back to Lecture 06 — Profiling: Where the Time Actually Goes](../lectures/06-profiling-where-the-time-actually-goes.md) · [Course Home](../index.md)
