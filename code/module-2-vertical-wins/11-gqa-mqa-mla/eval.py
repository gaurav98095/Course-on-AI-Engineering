"""The eval harness: turn "the answer looked right" into two real numbers.

Retrieval recall@k: is the right page even in the retrieved set?
Answer quality: does the generated answer contain what it must, and cite
the right page?

Needs a confirmed answer key first -- run build_eval_set.py once.

Run:  python eval.py                                              # recall@k only, fast
      python eval.py --generate                                   # + bf16 answer quality
      python eval.py --generate --checkpoint qwen3-vl-8b-gptq-4bit # same eval, quantized generator
"""

import argparse
import json
import re
from pathlib import Path

from rag import Generator, Retriever

EVAL_SET = Path("eval_set.json")
CITE_RE = re.compile(r"\[([\w-]+)\s+p\.(\d+)\]")


def load_eval_set() -> list[dict]:
    items = json.loads(EVAL_SET.read_text())
    missing = [it["question"] for it in items if it.get("expected_page") is None]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(items)} question(s) still have no confirmed answer -- "
            f"run build_eval_set.py first. First unconfirmed: {missing[0]!r}"
        )
    return items


def recall_at_k(retriever: Retriever, items: list[dict], k_text: int) -> float:
    hits = 0
    for item in items:
        hits_t, _ = retriever(item["question"], k_text=k_text)
        pages = {(t["doc"], t["page"]) for t in hits_t}
        if (item["expected_doc"], item["expected_page"]) in pages:
            hits += 1
    return hits / len(items)


def answer_quality(retriever: Retriever, generator: Generator, items: list[dict]) -> dict:
    term_hits, cite_hits = 0, 0
    for item in items:
        hits_t, hits_i = retriever(item["question"])
        answer, _, _ = generator(item["question"], hits_t, hits_i)
        lowered = answer.lower()

        if all(term.lower() in lowered for term in item["required_terms"]):
            term_hits += 1

        cited = {(doc, int(page)) for doc, page in CITE_RE.findall(answer)}
        if (item["expected_doc"], item["expected_page"]) in cited:
            cite_hits += 1

    n = len(items)
    return {"term_coverage": term_hits / n, "citation_accuracy": cite_hits / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="top-k for recall@k (default 4)")
    ap.add_argument("--generate", action="store_true",
                    help="also run full retrieve+generate and score answer quality")
    ap.add_argument("--checkpoint", default=None,
                    help="path to a GPTQModel-quantized checkpoint (default: stock bf16)")
    args = ap.parse_args()

    items = load_eval_set()
    retriever = Retriever()

    r_at_k = recall_at_k(retriever, items, args.k)
    print(f"retrieval recall@{args.k}: {r_at_k:.2f}  ({len(items)} questions)")

    if args.generate:
        label = args.checkpoint or "bf16"
        print(f"\ngenerating with: {label} ...")
        generator = Generator(quantized_path=args.checkpoint)
        scores = answer_quality(retriever, generator, items)
        print(f"required-term coverage: {scores['term_coverage']:.2f}")
        print(f"citation accuracy:      {scores['citation_accuracy']:.2f}")


if __name__ == "__main__":
    main()
