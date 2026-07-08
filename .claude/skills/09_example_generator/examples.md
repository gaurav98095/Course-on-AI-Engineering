# Examples

## Example 1

Input:

- Lesson on KV cache size
- Mental model: writing down partial results instead of recomputing

Output:

- E1 (trivial, intuition): 1 layer, 1 KV head, head dim 4, FP16 — one token costs 16 bytes of cache; count it by hand.
- E2 (running, math + code): the course model at 4k context, batch 8 — compute the gigabytes on paper, then confirm with `torch.cuda.memory_allocated()`.
- E3 (edge case, exercise): batch 64 at 32k context — the cache alone exceeds the GPU; this is why long-context serving is a memory problem, not a compute problem.

## Example 2

Input:

- Lesson on continuous batching

Output:

- E1 (trivial): two requests, one finishes at token 10, one at token 500 — static batching holds the finished slot hostage for 490 steps.
- E2 (running): 8 slots, Poisson arrivals — compute utilization for static vs continuous batching in a 20-line simulation.
- E3 (counterexample): all requests identical length — continuous batching wins nothing, showing the gain comes from length variance, not magic.
