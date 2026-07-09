"""Load a GPTQ- or AWQ-quantized model, measure real decode speed and
memory, and check the same Lecture 01 question still gets answered right.

This closes the loop Lecture 07 left open: does a *purpose-built* quantized
kernel actually deliver the speedup the roofline model promises (Math Deep
Dive 07's "c approx 0" case), and does answer quality survive quantization
at all?

Run:  python benchmark_quantized.py --path qwen3-vl-8b-gptq-4bit
      python benchmark_quantized.py --path qwen3-vl-8b-awq-4bit
"""

import argparse
import time

import torch
from gptqmodel import GPTQModel
from transformers import AutoProcessor

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
QUESTION = "Why does an aircraft stall at the critical angle of attack?"
N_TOKENS = 80


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="folder saved by quantize_gptq_awq.py")
    args = ap.parse_args()

    print(f"loading quantized model from {args.path} ...")
    model = GPTQModel.load(args.path)
    processor = AutoProcessor.from_pretrained(MODEL)

    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"weight memory (state dict): {weight_bytes / 2**30:.2f} GiB")

    messages = [{"role": "user", "content": [{"type": "text", "text": QUESTION}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    model.generate(**inputs, max_new_tokens=10)   # warmup
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=N_TOKENS)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    answer = model.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"\ngenerated {n_new} tokens in {elapsed:.2f}s -> {n_new / elapsed:.1f} tok/s")
    print(f"peak GPU memory during generation: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"\n--- quality spot check: same question as Lecture 01 ---\n{answer}")


if __name__ == "__main__":
    main()
