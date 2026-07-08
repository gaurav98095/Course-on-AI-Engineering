"""Number formats, bit by bit: watch real weight values round to their
nearest representable neighbor in fp32, fp16, bf16, and int8.

No model needed today — this is about the formats themselves, on a
handful of numbers small enough to check by hand.

Run:  python bit_formats.py
"""

import struct

import torch

VALUES = [3.14159265, 0.0001234, 70000.0, -1.5, 1.0 / 3.0]


def bits(x: float, dtype: torch.dtype) -> str:
    t = torch.tensor([x], dtype=dtype)
    raw = t.view(torch.uint8 if t.element_size() == 1 else
                torch.int16 if t.element_size() == 2 else torch.int32)
    n = raw.item() & (2 ** (8 * t.element_size()) - 1)
    return format(n, f"0{8 * t.element_size()}b")


def roundtrip(x: float, dtype: torch.dtype) -> float:
    return torch.tensor([x], dtype=dtype).float().item()


def main() -> None:
    print(f"{'value':>14} {'fp32 -> value':>16} {'fp16 -> value':>16} "
          f"{'bf16 -> value':>16} {'int8* -> value':>16}")
    for v in VALUES:
        fp16 = roundtrip(v, torch.float16)
        bf16 = roundtrip(v, torch.bfloat16)
        # int8*: a toy symmetric quantizer, scale chosen per-value just to
        # show the rounding step -- real quantizers pick one scale per
        # tensor (or per channel), covered in Step 2.
        scale = abs(v) / 127 if v != 0 else 1.0
        q = round(v / scale)
        int8 = q * scale
        print(f"{v:>14.6f} {fp16:>16.6f} {bf16:>16.6f} {int8:>16.6f}")

    print("\n--- same three values, their actual stored bits ---")
    for v in VALUES[:3]:
        print(f"{v:>14.6f}  fp16={bits(v, torch.float16)}  bf16={bits(v, torch.bfloat16)}")

    print("\n--- the range problem: a value fp16 cannot represent at all ---")
    big = 70000.0   # fp16's max finite value is 65504
    print(f"{big} in fp16 -> {roundtrip(big, torch.float16)}  (overflowed to inf!)")
    print(f"{big} in bf16 -> {roundtrip(big, torch.bfloat16)}  (bf16 kept fp32's exponent range)")


if __name__ == "__main__":
    main()
