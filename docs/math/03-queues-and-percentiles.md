---
layout: default
title: "Math Deep Dive — Queues, Percentiles, and Why p95 Explodes"
---

# Math Deep Dive — Queues, Percentiles, and Why p95 Explodes

> **Supports:** [Lecture 03 — Load-Test It Until It Breaks](../lectures/03-load-test-it-until-it-breaks.md). Read the lecture first; this page assumes its story and mental model.

## The Idea in One Picture

A server is a shop with one till. Customers arrive at some rate; the till serves at some rate. The entire drama of load — flat throughput, ramping latency, exploding p95, the timeout cliff — falls out of the ratio between those two rates. This page derives all of it with high-school algebra.

## Notation

| Symbol | Meaning | In our system |
| --- | --- | --- |
| \\(S\\) | service time: one request, no queue | ~7 s (200-token answers, L40S) |
| \\(\mu\\) | service rate, \\(1/S\\) | ~8.5 requests/min |
| \\(\lambda\\) | arrival rate | whatever users inflict |
| \\(\rho\\) | utilization, \\(\lambda/\mu\\) | 0 → idle, 1 → saturated |
| \\(L\\) | avg. requests in the system (waiting + being served) | = \\(C\\) in a closed loop |
| \\(W\\) | avg. time a request spends in the system | what users feel |
| \\(W_q\\) | avg. time waiting in queue (excludes service) | \\(W - S\\) |
| \\(C\\) | virtual users in a closed-loop test | our sweep levels |

## Derivation

### Part 1 — Little's law, provable on a napkin

Watch the shop for a long window of \\(T\\) seconds and count **request-seconds** — one unit for each second one request spends inside — in two different ways.

*Counting by requests:* \\(\lambda T\\) requests pass through, each spending on average \\(W\\) seconds inside. Total: \\(\lambda T \, W\\).

*Counting by time:* at any instant there are on average \\(L\\) requests inside; integrate over the window. Total: \\(L\,T\\).

Same quantity, so:

\\[
L\,T = \lambda T\, W \quad\Longrightarrow\quad L = \lambda W
\\]

No assumptions about arrival patterns, service distributions, or scheduling. It holds for our toy server, for vLLM, and for the Kubernetes cluster in Module 3. That generality is why it's the one queueing fact worth memorizing.

### Part 2 — The closed loop: why our sweep was a straight line

Our load generator keeps exactly \\(C\\) requests in flight — each virtual user always has one outstanding. So \\(L = C\\), pinned by construction.

A single worker serves one at a time, so throughput can never exceed \\(\mu\\); with the loop keeping it always busy, throughput *equals* \\(\mu = 1/S\\). Little's law then forces:

\\[
W = \frac{L}{\lambda} = \frac{C}{1/S} = C \times S
\\]

Every row of the lecture's sweep table is this line: \\(C=8 \Rightarrow 8 \times 7 = 56\\) s, \\(C=16 \Rightarrow 112\\) s. And the "cliff" is just this line crossing the timeout: errors must begin at

\\[
C^\* = \frac{T_{\text{timeout}}}{S} = \frac{120}{7} \approx 17
\\]

— which is why the table shows clean runs at 16 and carnage at 32.

### Part 3 — The open world: M/M/1 and the hockey stick

Real traffic doesn't wait politely. Model arrivals as random (Poisson) at rate \\(\lambda\\), service times exponential with mean \\(1/\mu\\), one server: the textbook **M/M/1** queue. Its stationary analysis (balance equations between states "n requests in system") yields a famously compact result:

\\[
W \;=\; \frac{1}{\mu - \lambda} \;=\; \frac{S}{1-\rho}
\\]

Read the denominator. \\(W\\) doesn't grow linearly with load — it grows like \\(1/(1-\rho)\\):

| \\(\rho\\) (utilization) | \\(W\\) (mean response) | vs. idle |
| --- | --- | --- |
| 0.50 | \\(2S\\) = 14 s | 2× |
| 0.80 | \\(5S\\) = 35 s | 5× |
| 0.90 | \\(10S\\) = 70 s | 10× |
| 0.95 | \\(20S\\) = 140 s | 20× |
| 0.99 | \\(100S\\) = 700 s | 100× |

> The last 10% of utilization costs more latency than the first 90%. "Run the GPUs hot" and "keep p95 low" are mathematically opposed — capacity planning (Lecture 28) is the art of choosing the operating point.

### Part 4 — Percentiles: why p95 is ~3× the mean near saturation

For M/M/1, the response time is exponentially distributed with mean \\(W\\). The probability a request takes longer than \\(t\\) is \\(e^{-t/W}\\), so the \\(p\\)-th percentile solves \\(e^{-t/W} = 1-p\\):

\\[
t_p = W \ln\!\frac{1}{1-p}
\qquad\Rightarrow\qquad
t_{95} = W\ln 20 \approx 3W, \quad t_{99} = W \ln 100 \approx 4.6W
\\]

So at \\(\rho = 0.9\\) (mean 70 s), p95 ≈ **3.5 minutes** and p99 ≈ **5.4 minutes** — from a server whose *service time is 7 seconds*. When an SLO says "p99 under 2 s", it is implicitly dictating a maximum \\(\rho\\), hence a minimum fleet size, hence a budget. That chain of implication is Module 3's capacity math.

## Worked Example

Our shop: \\(S = 7\\) s, so \\(\mu = 8.57\\)/min. The support team of 40 asks one question every 5 minutes each: \\(\lambda = 8\\)/min.

- Utilization: \\(\rho = 8 / 8.57 = 0.93\\).
- Mean response: \\(W = S/(1-\rho) = 7 / 0.067 \approx 105\\) s.
- p95: \\(\approx 3 \times 105 \approx 5.2\\) min. p99: \\(\approx 4.6 \times 105 \approx 8\\) min.
- And with a 120 s timeout, roughly \\(e^{-120/105} \approx 32\%\\) of requests **never return at all**.

One till cannot host this launch. The fixes, in the order this course teaches them: serve more requests per second on the same GPU (Module 2 — batching, kernels, engines: a ~100× lever), then more GPUs behind a queue-aware autoscaler (Module 3).

## Where the Assumptions Break

**Closed-loop tests hide the explosion.** Our generator only ever allows \\(C\\) in flight — when the server slows, arrivals slow with it. That's why our sweep showed a *linear* ramp, never the \\(1/(1-\rho)\\) hockey stick. Real users are open-loop. Worse, the standard sin — **coordinated omission** — is measuring latency only for the requests you managed to send, forgetting the ones delayed *by the queue you created*. Gil Tene's talk (lecture resources) is the canonical treatment.

**LLM service times aren't exponential — or even constant.** They're roughly linear in output length, so the distribution mirrors your answer-length mix, and long generations hog the single server (a form of head-of-line blocking). M/M/1's exact constants shift; its *shape* — the \\(1/(1-\rho)\\) blow-up — survives.

**One server, for now.** With \\(k\\) servers (M/M/k, i.e., continuous batching slots or a GPU fleet), waits drop dramatically at equal \\(\rho\\) — pooling is powerful, and it's exactly what continuous batching (Lecture 13) exploits inside a single GPU.

## Common Mistakes

- **Averaging percentiles.** The mean of two p95s is not the p95 of the merged traffic. Aggregate raw latencies (or histograms), then take percentiles.
- **Reporting capacity without a percentile and a load.** "Handles 8 req/min" is true at p50=7 s and at p50=110 s. Meaningless alone.
- **Treating timeouts as latency.** A timed-out request has no latency; it's an error. Mixing them into the latency array (or silently dropping them) both distort the truth — report error rate alongside percentiles, always.
- **Load-testing over the internet and blaming the server.** Test next to the server first (we did), then add the network back deliberately, as its own measured term.

---

[← Back to Lecture 03 — Load-Test It Until It Breaks](../lectures/03-load-test-it-until-it-breaks.md) · [Course Home](../index.md)
