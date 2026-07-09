"""Number formats, bit by bit: watch real numbers round to their nearest
representable neighbor in fp16, bf16, and int8.

Two separate demos on purpose. Part 1 uses values chosen to exercise
fp16/bf16's range and precision trade-off, including one value (70000)
that only exists to break fp16's range -- it would never appear in a real
weight tensor. Part 2 uses realistic weight-sized values for the int8
demo, because a shared-scale quantizer's whole point is meaningless if
one wild outlier is allowed to set the scale for everything else.

Run:  python bit_formats.py
"""

import torch

# Part 1: chosen to exercise range vs precision. 70000 exists ONLY to show
# fp16's overflow -- never a realistic weight value.
RANGE_PRECISION_VALUES = [3.14159265, 0.0001234, 70000.0, -1.5, 1.0 / 3.0]

# Part 2: realistic weight magnitudes for one linear layer's row -- this is
# what a shared-scale int8 quantizer is actually built for.
WEIGHT_LIKE_VALUES = [0.0421, -0.3180, 0.0037, 1.2040, -0.8120]


def bits(x: float, dtype: torch.dtype) -> str:
    t = torch.tensor([x], dtype=dtype)
    raw = t.view(torch.uint8 if t.element_size() == 1 else
                torch.int16 if t.element_size() == 2 else torch.int32)
    n = raw.item() & (2 ** (8 * t.element_size()) - 1)
    return format(n, f"0{8 * t.element_size()}b")


def roundtrip(x: float, dtype: torch.dtype) -> float:
    return torch.tensor([x], dtype=dtype).float().item()


def quantize_int8_shared_scale(values: list[float]) -> None:
    """One scale for the whole tensor -- exactly Lecture 07's naive approach."""
    scale = max(abs(v) for v in values) / 127
    print(f"shared scale = max(|values|) / 127 = {scale:.6f}\n")
    print(f"{'value':>10} {'int8 code':>10} {'dequant':>10} {'error':>10}")
    for v in values:
        code = max(-127, min(127, round(v / scale)))
        dequant = code * scale
        print(f"{v:>10.4f} {code:>10d} {dequant:>10.5f} {v - dequant:>+10.5f}")


def main() -> None:
    print("--- Part 1: fp16 vs bf16 -- range and precision ---\n")
    print(f"{'value':>14} {'fp16 -> value':>16} {'bf16 -> value':>16}")
    for v in RANGE_PRECISION_VALUES:
        fp16 = roundtrip(v, torch.float16)
        bf16 = roundtrip(v, torch.bfloat16)
        print(f"{v:>14.8f} {fp16:>16.8f} {bf16:>16.8f}")

    print("\n--- the first two values, their actual stored bits ---")
    for v in RANGE_PRECISION_VALUES[:2]:
        print(f"{v:>14.6f}  fp16={bits(v, torch.float16)}  bf16={bits(v, torch.bfloat16)}")

    print("\n--- the range problem: a value fp16 cannot represent at all ---")
    big = 70000.0   # fp16's max finite value is 65504
    print(f"{big} in fp16 -> {roundtrip(big, torch.float16)}  (overflowed to inf!)")
    print(f"{big} in bf16 -> {roundtrip(big, torch.bfloat16)}  "
          "(bf16 kept fp32's exponent range -- but is now coarser: not exact)")

    print("\n--- Part 2: int8, one shared scale, realistic weight values ---\n")
    quantize_int8_shared_scale(WEIGHT_LIKE_VALUES)


if __name__ == "__main__":
    main()
