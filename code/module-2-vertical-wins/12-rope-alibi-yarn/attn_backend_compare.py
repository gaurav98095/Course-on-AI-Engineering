"""Reload the course model with a different attention backend and re-run
Lecture 01's own stall question, timing TTFT specifically -- prefill on a
~1,800-token prompt is exactly where a fused attention kernel's smaller
HBM footprint should show up as real wall-clock time.

One backend per process, same reason Lecture 07 ran bf16/int8/int4
separately -- you can't hold multiple 8B-parameter copies on one GPU.

Run:  python attn_backend_compare.py --impl eager
      python attn_backend_compare.py --impl sdpa
      python attn_backend_compare.py --impl flash_attention_2   # needs: pip install flash-attn --no-build-isolation
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from rag import Retriever

GENERATOR = "Qwen/Qwen3-VL-8B-Instruct"
CORPUS = Path("corpus")
QUESTION = "Why does an aircraft stall at the critical angle of attack?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    args = ap.parse_args()

    print(f"loading {GENERATOR} with attn_implementation={args.impl!r} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        GENERATOR, dtype=torch.bfloat16, device_map="auto", attn_implementation=args.impl,
    )
    processor = AutoProcessor.from_pretrained(GENERATOR)

    retriever = Retriever()
    hits_t, hits_i = retriever(QUESTION)
    excerpts = "\n\n".join(f"[{t['doc']} p.{t['page']}] {t['text']}" for t in hits_t)
    content = [{"type": "image", "image": str(CORPUS / "images" / m["file"])} for m in hits_i]
    content.append({"type": "text", "text": f"Manual excerpts:\n{excerpts}\n\nQuestion: {QUESTION}"})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    # prefill only: generate exactly one new token, so the timed region is ~all prefill
    model.generate(**inputs, max_new_tokens=1)   # warmup -- pays kernel compilation once
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.generate(**inputs, max_new_tokens=1)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    print(f"\nprompt tokens: {inputs['input_ids'].shape[1]}")
    print(f"TTFT (prefill, attn_implementation={args.impl}): {ttft * 1000:.1f} ms")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
