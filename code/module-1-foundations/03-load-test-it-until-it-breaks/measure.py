"""First benchmark: TTFT, TPOT, tokens/sec, and VRAM for our RAG — single stream.

This table is the baseline the whole course optimizes. Save the output.

Run:  python measure.py
"""

import time

import torch
from threading import Thread
from transformers import TextIteratorStreamer

from rag import Generator, Retriever

QUESTIONS = [
    "Why does an aircraft stall at the critical angle of attack?",
    "How does the attitude indicator work?",
    "What is the difference between induced drag and parasite drag?",
    "How does the pitot-static system feed the airspeed indicator?",
    "What happens to lift when flaps are extended?",
]


@torch.inference_mode()
def timed_answer(gen: Generator, retrieve: Retriever, question: str) -> dict:
    hits_t, hits_i = retrieve(question)

    content = [{"type": "image", "image": f"corpus/images/{m['file']}"} for m in hits_i]
    excerpts = "\n\n".join(f"[{t['doc']} p.{t['page']}] {t['text']}" for t in hits_t)
    content.append({"type": "text", "text": f"Manual excerpts:\n{excerpts}\n\nQuestion: {question}"})
    messages = [{"role": "user", "content": content}]
    inputs = gen.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(gen.model.device)

    streamer = TextIteratorStreamer(gen.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = Thread(target=gen.model.generate,
                    kwargs=dict(**inputs, max_new_tokens=300, streamer=streamer))

    torch.cuda.synchronize()          # CUDA is async: without this, timers lie
    t0 = time.perf_counter()
    thread.start()
    ttft, n_out = None, 0
    for _ in streamer:                # first yielded chunk = first token has arrived
        if ttft is None:
            ttft = time.perf_counter() - t0
        n_out += 1
    thread.join()
    total = time.perf_counter() - t0

    return {
        "prompt_tokens": inputs["input_ids"].shape[1],
        "new_tokens": n_out,
        "ttft_s": ttft,
        "tpot_ms": (total - ttft) / max(n_out - 1, 1) * 1000,
        "tok_per_s": n_out / total,
        "total_s": total,
    }


def main() -> None:
    retrieve, gen = Retriever(), Generator()

    print("warmup (compiling kernels, filling caches — never time the first run)...")
    timed_answer(gen, retrieve, QUESTIONS[0])
    torch.cuda.reset_peak_memory_stats()

    rows = [timed_answer(gen, retrieve, q) for q in QUESTIONS]

    print(f"\n{'prompt':>7} {'new':>5} {'TTFT s':>7} {'TPOT ms':>8} {'tok/s':>7} {'total s':>8}")
    for r in rows:
        print(f"{r['prompt_tokens']:>7} {r['new_tokens']:>5} {r['ttft_s']:>7.2f} "
              f"{r['tpot_ms']:>8.1f} {r['tok_per_s']:>7.1f} {r['total_s']:>8.2f}")

    mean = lambda k: sum(r[k] for r in rows) / len(rows)
    print(f"\nmean: TTFT {mean('ttft_s'):.2f}s | TPOT {mean('tpot_ms'):.1f}ms | {mean('tok_per_s'):.1f} tok/s")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print(f"GPU: {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
