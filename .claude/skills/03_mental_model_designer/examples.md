# Examples

## Example 1

Topic: Roofline model

Primary model: Chef and conveyor belt

Why it works:

- The chef is compute (FLOPs), the belt is memory bandwidth (bytes)
- A dish is compute-bound when the chef is the bottleneck, memory-bound when the belt is
- Arithmetic intensity is dishes cooked per ingredient delivered — one number says which regime you are in

## Example 2

Topic: PagedAttention

Primary model: Hotel rooms instead of reserved floors

Why it works:

- Reserving a whole floor per guest (contiguous KV allocation) wastes every unused room
- Handing out rooms one at a time from a shared pool (blocks) wastes almost nothing
- The front-desk ledger (block table) remembers which rooms belong to which guest

## Example 3

Topic: Speculative decoding

Primary model: Junior drafts, senior signs off

Why it works:

- A fast junior writer drafts several sentences ahead
- The senior editor checks the whole draft in one read — parallel, so nearly free
- Wherever the senior disagrees, the draft is cut and generation resumes from there; the output is exactly what the senior alone would have written
