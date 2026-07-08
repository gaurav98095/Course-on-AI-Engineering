# Examples

## Example 1

Issue:

- The math section introduces `d_head` without defining head dimension.

Fix:

- Add a notation block before the equation and define every dimension symbol explicitly.

## Example 2

Issue:

- A benchmark table reports tokens/sec without naming the GPU, batch size, or precision.

Fix:

- Add a hardware-and-settings line above every benchmark table; numbers without context teach nothing.

## Example 3

Issue:

- The timing code measures the first run, which includes compilation and cache warmup.

Fix:

- Add a warmup call before the timed loop and a sentence explaining what the first run silently includes.
