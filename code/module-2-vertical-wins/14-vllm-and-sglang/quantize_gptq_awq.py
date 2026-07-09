"""Quantize the course model with GPTQ or AWQ, calibrated on our own corpus.

Unlike Lecture 07's naive round-to-nearest, both of these look at real data
before deciding how to round -- GPTQ compensates each weight's rounding
error into the weights quantized after it; AWQ finds and protects the
small number of "salient" weight channels that matter most to the model's
real outputs. Same GPTQModel API for both; only the config class changes.

Calibration data: our own FAA manual chunks from Lecture 01 -- quantize the
model against the exact kind of text it will actually be asked about.

Run:  python quantize_gptq_awq.py --method gptq
      python quantize_gptq_awq.py --method awq
"""

import argparse
import json
from pathlib import Path

from gptqmodel import AWQConfig, GPTQConfig, GPTQModel

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CORPUS = Path("corpus/chunks.json")   # from Lecture 01's ingest.py
N_CALIBRATION = 256                   # samples; more is slower, not always better


def load_calibration_text() -> list[str]:
    if not CORPUS.exists():
        raise FileNotFoundError(
            f"{CORPUS} not found -- run ingest.py first (Lecture 01) so we "
            "calibrate on our own manual, not a generic dataset."
        )
    chunks = json.loads(CORPUS.read_text())["texts"]
    texts = [c["text"] for c in chunks]
    return texts[:N_CALIBRATION]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["gptq", "awq"], required=True)
    ap.add_argument("--bits", type=int, default=4)
    args = ap.parse_args()

    calibration = load_calibration_text()
    print(f"calibrating on {len(calibration)} real chunks from our own corpus")

    if args.method == "gptq":
        # sym=True, group_size=128: quantize in blocks of 128 weights, one
        # shared scale per block -- finer-grained than Lecture 07's
        # one-scale-per-tensor naive int8/int4.
        quant_config = GPTQConfig(bits=args.bits, group_size=128)
    else:
        quant_config = AWQConfig(bits=args.bits, group_size=128)

    out_path = f"qwen3-vl-8b-{args.method}-{args.bits}bit"
    print(f"loading {MODEL} ...")
    model = GPTQModel.load(MODEL, quant_config)

    print(f"quantizing with {args.method.upper()} ({args.bits}-bit) -- "
          f"this walks the model layer by layer, expect several minutes...")
    model.quantize(calibration, batch_size=1)

    model.save(out_path)
    print(f"\nsaved to {out_path}/")


if __name__ == "__main__":
    main()
