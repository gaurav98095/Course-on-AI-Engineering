# Examples

## Example 1

Input: "Design a lesson for the KV cache."

Output:

- Prerequisites: self-attention, prefill vs decode, basic PyTorch
- Objectives: explain why decode recomputes nothing, derive the KV cache size formula, predict the memory of a real deployment before running it
- Duration: 45 minutes
- Difficulty: intermediate

## Example 2

Input: "Turn quantization into a two-lecture module."

Output:

- Lecture 1: number formats (FP16, FP8, INT8, INT4), what precision buys and what it costs, weight-only vs weight-and-activation
- Lecture 2: GPTQ and AWQ in practice — quantize the course model, re-run the benchmark, prove quality held with an eval

## Example 3

Input: "Outline the prerequisites for PagedAttention."

Output:

- KV cache math
- Why contiguous cache allocation fragments memory
- Continuous batching
- Virtual memory intuition (pages, blocks, lookup tables)
