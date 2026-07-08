---
layout: default
title: "Math Deep Dive — Sampling Rate and Smoothing a Live Signal"
---

# Math Deep Dive — Sampling Rate and Smoothing a Live Signal

> **Supports:** [Lecture 01b — GPU Vitals: Watching What You Built](../lectures/01b-gpu-vitals.md). Read the lecture first; this page assumes its story and mental model.

## The Idea in One Picture

A vitals monitor doesn't see the GPU continuously — it takes snapshots. Between snapshots, anything can happen and vanish without a trace. This page makes precise how fast you need to sample to trust what you see, and the simple trick real dashboards use to summarize a stream of samples without storing every one of them.

## Notation

| Symbol | Meaning | Our system |
| --- | --- | --- |
| \\(\Delta t\\) | sampling interval | 1 s (`dmon` default), 0.05–1 s (`gpu_vitals.py`) |
| \\(\tau\\) | duration of the event you're trying to see | ~30 ms (one decode step) |
| \\(f_s\\) | sampling frequency, \\(1/\Delta t\\) | 1 Hz default |
| \\(x_t\\) | the raw signal's true value at time \\(t\\) | utilization %, power, etc. |
| \\(\hat{x}_n\\) | the smoothed (EMA) estimate after \\(n\\) samples | what a dashboard displays |
| \\(\alpha\\) | EMA smoothing factor, \\(0 < \alpha \le 1\\) | tunable |

## Derivation

### Part 1 — Why a spike can vanish entirely, not just blur

Consider the simplest possible case: a signal that's 0 almost always, and jumps to some value \\(V\\) for a single short burst of duration \\(\tau\\), then returns to 0. Your sampler takes a snapshot every \\(\Delta t\\) seconds, instantaneously.

If \\(\tau < \Delta t\\) (the burst is shorter than the gap between samples), there's no guarantee any sample lands *during* the burst at all. The probability a single sample happens to catch it, for a burst arriving at a uniformly random offset within one sampling interval, is:

\\[
P(\\text{catch it}) = \\frac{\\tau}{\\Delta t}
\\]

Plug in our numbers: \\(\tau = 30\\text{ms}\\), \\(\Delta t = 1000\\text{ms}\\):

\\[
P = \\frac{30}{1000} = 0.03
\\]

**A 1-second sampler has only a 3% chance of ever recording a single isolated 30ms spike.** This is not a rounding error — it's the difference between "this optimization changed nothing" and "I never had a chance to see it" being visually indistinguishable on your dashboard.

### Part 2 — Why our GPU trace still looked smooth (and correctly so)

Section 1 assumed *one* isolated burst. Decode isn't one burst — it's dozens of them back to back, one per token, for the whole duration of a request. Averaging Part 1's logic over many bursts within one sample window changes the conclusion completely: instead of "might miss it entirely," you get "will average it," because a 1-second window during active decoding contains roughly:

\\[
\\frac{\\Delta t}{\\tau} = \\frac{1000\\text{ms}}{30\\text{ms}} \\approx 33 \\text{ decode steps}
\\]

Each `dmon` reading during generation isn't a snapshot of one step — it's an **implicit average** over about 33 of them. That's exactly why Lecture 01b's dashboard showed one smooth hump per *request* rather than 33 jagged spikes: the sampling rate was too coarse to resolve individual tokens, but more than fine enough to resolve whole requests (which last seconds, not milliseconds).

### Part 3 — The rule of thumb, stated properly

Generalizing Parts 1 and 2: to reliably resolve an event of duration \\(\tau\\) (see it as a clean, undistorted shape rather than a blur or a coin-flip), you need your sampling interval comfortably smaller than the event:

\\[
\\Delta t \\ll \\tau \\qquad\\Longleftrightarrow\\qquad f_s \\gg \\frac{1}{\\tau}
\\]

This is a plain-language version of the Nyquist sampling principle from signal processing: to resolve a feature, sample several times *within* it, not once every few of it. For our numbers, resolving individual 30ms decode steps needs \\(\Delta t\\) well under 30ms — exactly why Exercise 2 in the lecture asks you to try `--interval 0.05` (50ms, finally fast enough to start seeing individual steps) instead of the 1-second default.

### Part 4 — Smoothing without storing everything: the exponential moving average

Real dashboards face the opposite problem too: sampling *fast* produces a noisy, jittery signal that's hard to read, and storing every raw sample forever is wasteful. The standard fix is the exponential moving average — a running estimate updated one sample at a time, with no history required:

\\[
\\hat{x}_n = \\alpha\\, x_n + (1-\\alpha)\\, \\hat{x}_{n-1}
\\]

Each new estimate is a blend: mostly the previous estimate, nudged toward the newest raw sample by \\(\alpha\\). Unroll the recursion one step to see why this counts as "moving average" at all:

\\[
\\hat{x}_n = \\alpha x_n + \\alpha(1-\\alpha) x_{n-1} + \\alpha(1-\\alpha)^2 x_{n-2} + \\cdots
\\]

Every past sample contributes forever, but its weight shrinks geometrically — recent samples dominate, old ones fade, and you never need to store more than one running number (\\(\hat{x}_{n-1}\\)) to compute the next.

## Worked Example

Take three consecutive raw utilization readings during a burst: \\(x_1 = 10\\%\\), \\(x_2 = 90\\%\\), \\(x_3 = 15\\%\\) (a single noisy spike), with \\(\alpha = 0.3\\) and \\(\hat{x}_0 = 10\\%\\) (starting at the first value):

\\[
\\hat{x}_1 = 0.3(10) + 0.7(10) = 10.0
\\]
\\[
\\hat{x}_2 = 0.3(90) + 0.7(10.0) = 27 + 7 = 34.0
\\]
\\[
\\hat{x}_3 = 0.3(15) + 0.7(34.0) = 4.5 + 23.8 = 28.3
\\]

The raw signal swung wildly (10 → 90 → 15); the smoothed estimate rose gently to 34 and eased back to 28.3 — visible, but not jarring. Push \\(\alpha\\) toward 1 and the EMA tracks the raw signal almost exactly (noisy, responsive); push it toward 0 and it barely moves (smooth, laggy). Choosing \\(\alpha\\) is choosing that trade-off on purpose, which is exactly what Lecture 27's production dashboards do at scale.

## Where the Assumptions Break

**Part 1's "uniformly random offset" is an idealization.** If your sampler and your workload happen to share any periodicity (e.g., both tied to some fixed clock), samples can land *systematically* in or out of phase with events, giving a biased picture rather than a randomly-missed one. This is a real failure mode in production monitoring, not just a textbook curiosity.

**EMA has memory, which can mislead after a real regime change.** If utilization genuinely jumps from 10% to 90% and *stays there*, the EMA takes several samples to catch up (per the geometric-decay formula in Part 4) — during that catch-up window, the dashboard is still showing a "warming up" number, not the new reality. Don't read an EMA mid-transition as a stable value.

**Higher sampling rate isn't free.** Part 3 says "sample faster to resolve short events," but polling overhead and log volume both grow with \\(f_s\\) — at extreme rates, the act of measuring can perturb what you're measuring (the lecture's own "Where It Breaks" section). Pick the slowest rate that still satisfies \\(\Delta t \\ll \\tau\\) for the event you actually care about, not the fastest rate available.

## Common Mistakes

- **Trusting a coarse trace to show short events.** A 1-second `dmon` trace showing "smooth" utilization during decode isn't evidence that decode is smooth — Part 2 shows it's an average of ~33 sub-events, not a direct observation of any one of them.
- **Confusing EMA smoothness with signal smoothness.** A heavily-smoothed (\\(\alpha\\) near 0) chart can visually hide real spikes that matter — smoothing trades visibility of noise for visibility of genuine short-lived signal, and you don't get both from one \\(\alpha\\).
- **Picking \\(\Delta t\\) by convention instead of by the event you're measuring.** "1 second" is `dmon`'s default, not a law — Part 3's inequality is the actual criterion, and different questions (request-level vs. token-level) need different sampling rates.

---

[← Back to Lecture 01b — GPU Vitals: Watching What You Built](../lectures/01b-gpu-vitals.md) · [Course Home](../index.md)
