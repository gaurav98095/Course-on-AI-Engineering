"""Profile one real RAG request end to end: name every millisecond.

Uses torch.profiler — works on any machine, no extra install required.
For the professional-tool path (nsys / ncu), see this lecture's README
and the "Step 2/3" commands in the lecture itself.

Run:  python profile_request.py
"""

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from rag import Generator, Retriever

QUESTION = "Why does an aircraft stall at the critical angle of attack?"
MAX_NEW_TOKENS = 100


def main() -> None:
    retrieve, gen = Retriever(), Generator()

    print("warmup (never profile a cold run — Lecture 01 already taught us why)...")
    hits_t, hits_i = retrieve(QUESTION)
    gen(QUESTION, hits_t, hits_i, max_new_tokens=10)
    torch.cuda.synchronize()

    print("profiling one full request...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False,
    ) as prof:
        with record_function("retrieval"):
            hits_t, hits_i = retrieve(QUESTION)
        with record_function("generation"):
            answer, n_in, n_out = gen(QUESTION, hits_t, hits_i, max_new_tokens=MAX_NEW_TOKENS)

    print(f"\nprompt: {n_in} tokens -> generated: {n_out} tokens\n")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    prof.export_chrome_trace("trace.json")
    print("\nfull timeline saved to trace.json")
    print("open it at chrome://tracing (Chrome) or https://ui.perfetto.dev (any browser)")


if __name__ == "__main__":
    main()
