# Examples

## Example 1

Topic: KV cache size

Sequence:

- Explain why decode must remember every past token's keys and values
- Define layers, KV heads, head dimension, sequence length, batch, bytes per element
- Write the cache size product
- Define each symbol
- Work the number for the course model at 4k context, batch 8 — then check it against `nvidia-smi`

## Example 2

Topic: Arithmetic intensity and the roofline

Sequence:

- Explain compute-bound vs memory-bound with the chef-and-belt picture
- Define FLOPs, bytes moved, arithmetic intensity, peak compute, peak bandwidth
- Derive the ridge point of the GPU
- Interpret why decode at batch 1 sits far left of the ridge and prefill sits right of it
- Work the number: matrix-vector vs matrix-matrix on the course GPU

## Example 3

Topic: Speculative decoding speedup

Sequence:

- Explain draft-then-verify with the junior/senior picture
- Define draft length, acceptance rate, draft and target model costs
- Derive expected tokens accepted per target forward pass
- Show the break-even acceptance rate for a given draft/target cost ratio
- Work a small example: 4 draft tokens at 70% acceptance
