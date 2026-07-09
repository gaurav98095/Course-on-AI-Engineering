---
layout: default
title: "Math Deep Dive — How Many Eval Questions Are Enough?"
---

# Math Deep Dive — How Many Eval Questions Are Enough?

> **Supports:** [Lecture 08b — Build the Eval Harness](../lectures/08b-build-the-eval-harness.md). Read the lecture first; this page assumes its eval set and its recall@k / answer-quality scores.

## The Idea in One Picture

Recall@k over \\(N\\) questions is a fraction — 9 out of 10, say, or 0.90. A fraction looks exact. It isn't. Every question you didn't ask is a question that could have gone either way, and with only 10 asked, the fraction you got is one of many you could plausibly have gotten by chance alone. This page puts a number on that "could have gone either way" — and answers the question every quantization comparison in this course has been dodging: is a 5-point recall difference between two models *real*, or is it noise from too small an eval set?

## Notation

| Symbol | Meaning | Our 10-question eval set |
| --- | --- | --- |
| \\(N\\) | number of eval questions | 10 |
| \\(x\\) | number answered correctly | e.g. 9 |
| \\(\hat p\\) | observed proportion, \\(x/N\\) | 0.90 |
| \\(SE\\) | standard error of \\(\hat p\\) | computed below |
| \\(z\\) | critical value for a confidence level | 1.96 for 95% |

## Derivation

### Part 1 — An eval score is a coin flip, repeated N times

Each question in the eval set has one of two outcomes: the right page came back in the top-k, or it didn't. That's a **Bernoulli trial** — a weighted coin flip, with true (unknown) probability \\(p\\) of landing "correct." Ask \\(N\\) independent questions and count the successes, and \\(x\\) follows a **binomial distribution**: \\(x \sim \text{Binomial}(N, p)\\). Recall@k, \\(\hat p = x/N\\), is our one *estimate* of the true \\(p\\) — not \\(p\\) itself. Two different eval sets of the same size, asking different (but equally fair) questions, would produce two different \\(\hat p\\) values even if the underlying system never changed.

### Part 2 — How uncertain is that estimate?

The variance of a binomial proportion has a closed form:

\\[
\text{Var}(\hat p) = \frac{p(1-p)}{N}
\\]

Since we don't know the true \\(p\\), we plug in our estimate \\(\hat p\\) — the standard **normal approximation** to the binomial, valid when \\(N\hat p\\) and \\(N(1-\hat p)\\) aren't tiny:

\\[
SE = \sqrt{\frac{\hat p(1-\hat p)}{N}}, \qquad \text{95% CI} = \hat p \pm 1.96 \cdot SE
\\]

Read the \\(SE\\) formula once, slowly: it has \\(N\\) in the *denominator*, under a square root. Quadruple your eval set and the uncertainty only halves. This is the single most important fact on this page — eval-set precision is expensive to buy.

### Part 3 — Plug in our own numbers

\\(N=10\\), \\(x=9\\) — the exact shape of a "looks great" result:

\\[
\hat p = 0.90, \qquad SE = \sqrt{\frac{0.90 \times 0.10}{10}} = 0.0949, \qquad \text{95% CI} = 0.90 \pm 0.186 = [0.71,\ 1.00]
\\]

*(All numbers on this page are computed directly, not typed by hand — verify them yourself with five lines of Python.)*

That interval spans from "one bad question away from a mediocre system" to "perfect." A recall@4 of 0.90 on a 10-question eval set is barely distinguishable, statistically, from 0.75. This is not a flaw in the math — it's an honest report of how little a 10-question sample actually pins down.

### Part 4 — The interval shrinks, slowly, with more questions

Same observed proportion (0.90), ten times the eval set:

| \\(N\\) | \\(x\\) | \\(\hat p\\) | \\(SE\\) | 95% CI |
| --- | --- | --- | --- | --- |
| 10 | 9 | 0.90 | 0.095 | [0.71, 1.00] |
| 100 | 90 | 0.90 | 0.030 | [0.84, 0.96] |

Ten times the questions, roughly one-third the interval width — \\(SE \propto 1/\sqrt N\\), so cutting the interval in half takes *four* times the questions, not two.

### Part 5 — How big does an eval set need to be to trust a comparison?

The real question this course keeps asking informally — "is GPTQ's answer quality actually the same as bf16's?" — is a comparison of two proportions, \\(p\_1\\) and \\(p\_2\\). The same variance idea extends directly: the uncertainty in a *difference* of two independent proportions is the sum of their variances. Requiring that a true gap \\(\delta = p\_1-p\_2\\) be detected most of the time (95% confidence, 80% power, the standard textbook bar) gives the two-proportion sample size formula:

\\[
n \;\approx\; \frac{\left(z\_{\alpha/2}\sqrt{2\bar p(1-\bar p)} + z\_{\beta}\sqrt{p\_1(1-p\_1)+p\_2(1-p\_2)}\right)^2}{\delta^2}, \qquad \bar p = \frac{p\_1+p\_2}{2}
\\]

— \\(n\\) questions *per model*, not total. Plug in two illustrative cases, \\(z\_{0.025}=1.96\\), \\(z\_{0.20}=0.84\\):

| Comparing | \\(\delta\\) | \\(n\\) per model | Total questions |
| --- | --- | --- | --- |
| 0.90 vs 0.75 (a big, obvious gap) | 0.15 | 100 | 200 |
| 0.90 vs 0.80 (a real but smaller gap) | 0.10 | 199 | 398 |

Our 10-question eval set could not reliably tell these two models apart even at the *larger* gap — it would need to be **10–20× bigger** before "GPTQ looks about the same as bf16" becomes a claim you can actually stand behind, rather than an educated guess dressed up as a number.

## Worked Example

Applying Part 3's formula to two hypothetical runs of `eval.py --generate` on our own 10-question set: bf16 answers 9/10 correctly (\\(\hat p=0.90\\), CI \\([0.71,1.00]\\)); a quantized checkpoint answers 8/10 (\\(\hat p=0.80\\), CI \\([0.55,1.00]\\)). The two intervals overlap almost entirely. **You cannot conclude quantization hurt quality from this result alone** — the observed 10-point drop is comfortably within the noise a 10-question eval set produces even when nothing changed. Per Part 5, you'd need on the order of 200 questions per model before a genuine 10-point gap becomes statistically visible.

## Where the Assumptions Break

**The normal approximation is worst exactly where we're using it.** \\(N=10\\) and \\(\hat p\\) near 1.0 is precisely the regime where the normal approximation to the binomial is least accurate — our own \\([0.71, 1.00]\\) interval is itself a rough estimate, not an exact one. The **Wilson score interval** handles small \\(N\\) and extreme \\(\hat p\\) properly and is the standard fix in practice; the qualitative conclusion here (the interval is huge) holds regardless of which formula computes it.

**Questions aren't always independent.** The binomial model assumes each question is an independent coin flip. If several eval questions probe the same underlying weakness (say, three different phrasings of the same stall question), a single failure mode can fail all three at once — inflating your effective sample size beyond what the raw \\(N\\) suggests, and making the true uncertainty even larger than this page's numbers show.

**A bigger eval set fixes sampling noise, not eval design.** No amount of \\(N\\) rescues an eval set whose questions don't actually test what you care about, or whose answer key is wrong. Math can only ever quantify the uncertainty *given* the eval set is asking the right things — Lecture 08b's `build_eval_set.py` step (a human confirms every answer) still matters more than raw size.

## Common Mistakes

- **Reporting recall@4 to two decimal places from 10 questions.** 0.90 implies far more precision than a 10-question sample supports — round to what the confidence interval actually justifies, or report the interval itself.
- **Declaring a winner from overlapping confidence intervals.** If bf16's and GPTQ's intervals overlap substantially, the honest conclusion is "no significant difference detected at this sample size," not "they're equal" and not "bf16 won."
- **Confusing "the eval set didn't catch a regression" with "there is no regression."** Absence of evidence at \\(N=10\\) is extremely weak evidence of absence — Part 5's 200-question figure is the actual bar for confidence, not a suggestion.

---

[← Back to Lecture 08b — Build the Eval Harness](../lectures/08b-build-the-eval-harness.md) · [Course Home](../index.md)
