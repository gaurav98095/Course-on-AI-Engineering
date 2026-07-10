---
layout: default
title: "Math Deep Dive — Pooling: Why One Queue Beats N, and Where the Backroom Runs Out"
---

# Math Deep Dive — Pooling: Why One Queue Beats N, and Where the Backroom Runs Out

> **Supports:** [Lecture 03b — Fix It at the API Layer, First](../lectures/03b-api-layer-concurrency.md). Read the lecture first; this page assumes its story and mental model.

<!-- MATH DELIMITERS: in the markdown SOURCE, always write inline math as \\(...\\) and display math as \\[...\\] —
     DOUBLE backslash, not single, and never $...$/$$...$$. Verify by rendering, not by re-reading the markdown. -->

## The Idea in One Picture

Lecture 03's shop had one till, so opening a second till should just... work twice as well. It does, but only in the way Little's law predicts, and only until the tills run out of shared backroom stock. This page derives both halves: the exact speedup from \\(N\\) tills sharing one line, and the exact arithmetic that caps \\(N\\).

## Notation

| Symbol | Meaning | In our system |
| --- | --- | --- |
| \\(S\\) | service time, one request, no queue | ~7 s (200-tok answers, L40S) |
| \\(N\\) | worker processes (tills) on the one GPU | 1, 2, 3, ... |
| \\(C\\) | concurrent virtual users (closed loop) | our sweep levels |
| \\(\mu_N\\) | pooled throughput ceiling, \\(N/S\\) | ~8.5/min at \\(N=1\\), ~17/min at \\(N=2\\) |
| \\(W\\) | mean time a request spends in the system | what users feel |
| \\(V\\) | total GPU memory | 48 GiB (L40S) |
| \\(v\\) | memory one worker needs (weights + working set) | ~16–17 GiB (8B model, bf16) |
| \\(N_{\max}\\) | largest \\(N\\) that fits in \\(V\\) | what we measure today |

## Derivation

### Part 1 — Little's law with \\(N\\) tills instead of one

Same closed loop as Lecture 03: \\(C\\) virtual users, each always with exactly one request in flight, so \\(L = C\\), pinned by construction.

With \\(N\\) tills sharing **one** dispatch queue — whichever till frees up next takes the next customer, which is exactly how LitServe routes requests across `workers_per_device` — all \\(N\\) tills stay busy as long as \\(C \geq N\\). Throughput is no longer capped at \\(1/S\\); it's capped at:

\\[
\mu_N = \frac{N}{S}
\\]

Little's law then gives the new wait:

\\[
W = \frac{L}{\mu_N} = \frac{C}{N/S} = \frac{C}{N}\times S
\\]

Same straight line as Lecture 03's math page, just with the slope divided by \\(N\\). Double the tills, halve the ramp. \\(C=16\\), \\(N=2\\): \\(W \approx 8 \times 7 = 56\\) s — half of the 110 s one till produced.

### Part 2 — Why a shared queue beats splitting into \\(N\\) separate lines

Here's the subtlety a naive reading of "open a second till" misses: it matters *which* queueing discipline you use.

**Split queues.** Customers commit to a line before knowing its length. Each till is now its own independent single-server queue serving \\(C/N\\) users. If till 1's line has 3 people and till 2's has 0, an unlucky customer stuck behind those 3 waits \\(3S\\) — even though till 2 sat idle the whole time.

**One shared queue.** Whichever till frees up next takes the next customer. That same unlucky customer, in a shared queue, is served by whichever till is free — often immediately, never waiting behind an idle neighbor.

A short concrete check: 2 tills, one arrival finds queue state (3 waiting, 0 waiting). Split-queue expected wait for a customer randomly assigned to a line: \\(\tfrac{1}{2}(3S) + \tfrac{1}{2}(0) = 1.5S\\). Shared-queue wait for the same arrival: \\(0\\) — send them straight to the idle till. Pooling didn't add capacity; it just stopped wasting the capacity that was already idle.

> This effect has a name — **Erlang C / M/M/N** — and it is the same reason a bank's one snaking line feeding several windows beats \\(N\\) separate lines at equal total load. LitServe gives you the shared-queue version automatically; nothing to configure.
{: .remember}

### Part 3 — The arithmetic that caps \\(N\\)

Each worker is its own OS process, running its own `setup()`, holding its own copy of the model's weights plus its own KV cache and activation memory. None of that is shared across workers — replication, not pooling, at the memory layer. So:

\\[
N_{\max} = \left\lfloor \frac{V}{v} \right\rfloor
\\]

Plug in our numbers: \\(V = 48\\) GiB, \\(v \approx 16\text{–}17\\) GiB (an 8B model's bf16 weights, plus enough headroom for one request's KV cache and activations):

\\[
N_{\max} = \left\lfloor \frac{48}{17} \right\rfloor = 2 \text{ comfortably; a third worker leaves almost no headroom}
\\]

This is exactly the ceiling Lecture 03b's Step 5 hits: \\(N=2\\) runs cleanly, \\(N=3\\) is at or past the edge on a 48 GiB card. A bigger card (say an 80 GiB A100/H100) pushes \\(N_{\max}\\) higher, but it's always a *ceiling*, never a knob you can turn indefinitely.

### Part 4 — Why this lever alone doesn't get you to 100×

Cost per answer is (GPU $/hr) ÷ (answers/hr). Since \\(N\\) workers all still run on the **one** GPU we're already renting, the hourly cost doesn't change as \\(N\\) grows — only throughput does:

\\[
\text{cost per answer} = \frac{\text{price/hr}}{\mu_N \times 60} = \frac{\text{price/hr}}{N \times (60/S)}
\\]

Cost per answer falls linearly in \\(N\\) — real, measurable savings, free up to \\(N_{\max}\\). But \\(N_{\max}\\) is small (2–3 on this card) because each worker pays the full weight cost again. Module 2's levers — quantization, FlashAttention, continuous batching — instead shrink \\(v\\) and \\(S\\) themselves, so one process can serve many requests without ever paying for a second full copy of the weights. That's a fundamentally larger budget than replication can reach, which is why the course spends eight lectures on it instead of stopping here.

## Worked Example

L40S, 48 GiB, \\(S = 7\\) s, 8B model at bf16 (\\(v \approx 17\\) GiB):

- \\(N=1\\) (Lecture 03): \\(\mu_1 = 8.57\\)/min, \\(W(C{=}16) = 16 \times 7 = 112\\) s.
- \\(N=2\\): \\(\mu_2 = 17.14\\)/min, \\(W(C{=}16) = \tfrac{16}{2}\times 7 = 56\\) s — half the wait, double the ceiling, same GPU bill.
- \\(N_{\max} = \lfloor 48/17 \rfloor = 2\\): a third worker has \\(48 - 3\times17 = -3\\) GiB to spare before even one request's KV cache and activations — the arithmetic already says no before you type the command.
- Cost per answer at \\(N=2\\) is half of \\(N=1\\)'s, at the same \\(\$/\\)hr — queueing didn't make answers cheaper (Lecture 03), but pooling *did*, right up to \\(N_{\max}\\).

## Where the Assumptions Break

**Round-robin isn't perfectly even.** Real answer lengths vary, so one till can still get temporarily unlucky with a long generation even inside a shared queue — the imbalance shrinks with pooling, it doesn't vanish.

**Compute contention isn't accounted for here.** \\(N\\) workers on one GPU also time-slice the same SMs. We ignored that because Lecture 03 measured under 1% compute utilization per request — nowhere near the compute ceiling at \\(N=2\\) or \\(3\\). At much higher \\(N\\) (were memory not the limit first), compute contention would eventually bind too.

**\\(v\\) isn't fixed.** It depends on precision, model size, and how much KV cache headroom you reserve per worker — Lecture 07 onward changes \\(v\\) directly, which is a second, independent way to raise \\(N_{\max}\\) beyond what this page assumes constant.

## Common Mistakes

- **Treating `workers_per_device` as free capacity.** It's bounded by VRAM division, not by magic; compute the ceiling before you configure it.
- **Confusing more workers with more GPUs.** This page's whole argument lives inside one rented GPU. Module 3's fleet autoscaling repeats the same pooling idea across many GPUs behind a real load balancer — a different \\(N\\), a different ceiling.
- **Assuming split queues and a shared queue give the same wait at equal load.** They don't — Part 2's arithmetic is the proof, and it's exactly the gap between a naive "N separate lines" reading of "add workers" and what LitServe actually does for you.

---

[← Back to Lecture 03b — Fix It at the API Layer, First](../lectures/03b-api-layer-concurrency.md) · [Course Home](../index.md)
