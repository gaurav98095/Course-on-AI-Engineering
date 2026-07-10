"""Lecture 08b's eval harness, pointed at a real serving engine instead of
an in-process model -- the same recall@k and answer-quality checks, run
against whatever vLLM or SGLang server client_vllm.py already talks to.

Retrieval is still local and unchanged (Lecture 01's Retriever never moved).
Only generation crosses the network, exactly like client_vllm.py.

Needs a confirmed answer key first -- run build_eval_set.py once, and a
running server (see client_vllm.py's docstring for the two commands).

Run:  python eval_vllm.py                          # recall@k only, fast, no server needed
      python eval_vllm.py --generate                # + answer quality via vLLM on :8000
      python eval_vllm.py --generate --port 30000    # same eval, against SGLang instead
"""

import argparse

from openai import OpenAI

from client_vllm import build_messages
from eval import answer_quality, load_eval_set, recall_at_k
from rag import GENERATOR, Retriever


class RemoteGenerator:
    """Same call signature as rag.py's Generator, answered over HTTP instead
    of in-process -- so eval.py's answer_quality() doesn't need to know or
    care which one it's holding."""

    def __init__(self, port: int, max_tokens: int = 400) -> None:
        self.client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        self.max_tokens = max_tokens

    def __call__(self, question: str, hits_t: list[dict], hits_i: list[dict],
                 user_image=None, max_new_tokens: int | None = None):
        messages = build_messages(question, hits_t, hits_i)
        response = self.client.chat.completions.create(
            model=GENERATOR, messages=messages,
            max_tokens=max_new_tokens or self.max_tokens,
        )
        answer = response.choices[0].message.content
        usage = response.usage
        n_in = usage.prompt_tokens if usage else 0
        n_out = usage.completion_tokens if usage else 0
        return answer, n_in, n_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="top-k for recall@k (default 4)")
    ap.add_argument("--generate", action="store_true",
                    help="also run full retrieve+generate over HTTP and score answer quality")
    ap.add_argument("--port", type=int, default=8000, help="8000 for vLLM, 30000 for SGLang")
    args = ap.parse_args()

    items = load_eval_set()
    retriever = Retriever()

    r_at_k = recall_at_k(retriever, items, args.k)
    print(f"retrieval recall@{args.k}: {r_at_k:.2f}  ({len(items)} questions)")

    if args.generate:
        print(f"\ngenerating via server on port {args.port} ...")
        generator = RemoteGenerator(port=args.port)
        scores = answer_quality(retriever, generator, items)
        print(f"required-term coverage: {scores['term_coverage']:.2f}")
        print(f"citation accuracy:      {scores['citation_accuracy']:.2f}")


if __name__ == "__main__":
    main()
