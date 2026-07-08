"""Load the course model at bf16, int8, and int4 -- with zero calibration --
and measure what's actually free (memory) versus what isn't automatically
free (speed).

bitsandbytes gives us this with one config flag each; no calibration data,
no algorithm to tune -- that's tomorrow's lecture (GPTQ/AWQ). Today we just
watch what a naive, un-optimized quantized kernel actually costs.

Run:  python quantize_and_measure.py --mode bf16
      python quantize_and_measure.py --mode int8
      python quantize_and_measure.py --mode int4
"""

import argparse
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT = ("Explain, in detail, how an aircraft's pitot-static system works, "
          "and why it can fail in icing conditions.")
N_TOKENS = 80


def load(mode: str):
    kwargs = dict(device_map="auto")
    if mode == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif mode == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif mode == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(mode)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, **kwargs)
    processor = AutoProcessor.from_pretrained(MODEL)
    return model, processor


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bf16", "int8", "int4"], required=True)
    args = ap.parse_args()

    print(f"loading {MODEL} in {args.mode} ...")
    model, processor = load(args.mode)

    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"weight memory (state dict): {weight_bytes / 2**30:.2f} GiB")

    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    # warmup -- never time a cold run, Lecture 01 already taught us why
    model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=N_TOKENS, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    print(f"\ngenerated {n_new} tokens in {elapsed:.2f}s -> {n_new / elapsed:.1f} tok/s")
    print(f"peak GPU memory during generation: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
