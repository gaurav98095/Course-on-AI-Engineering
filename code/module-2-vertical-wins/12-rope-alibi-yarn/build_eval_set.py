"""Bootstrap eval_set.json's answer key from your own verified retrieval run.

An eval set is only as good as its answer key -- and a hand-typed "correct
page" is a guess until someone actually checks it against the real index.
So instead of shipping a pre-filled key (which would be a guess about
*your* re-ingested corpus, not a verified fact), this script runs
retrieval for every question that still needs one, shows you the real
hits, and lets you confirm which page is actually correct. That's how a
real eval set gets built: a human looks once, and that becomes the
reference every future run is checked against.

Run:  python build_eval_set.py
"""

import json
from pathlib import Path

from rag import Retriever

EVAL_SET = Path("eval_set.json")


def main() -> None:
    items = json.loads(EVAL_SET.read_text())
    retriever = Retriever()
    confirmed = 0

    for item in items:
        if item.get("expected_page") is not None:
            confirmed += 1
            continue

        hits_t, _ = retriever(item["question"], k_text=5)
        print(f"\nQ: {item['question']}")
        for i, t in enumerate(hits_t):
            print(f"  [{i}] {t['doc']} p.{t['page']}: {t['text'][:90]}...")

        choice = input("Which index is the real answer? ('s' to skip): ").strip()
        if choice.lower() == "s":
            continue

        chosen = hits_t[int(choice)]
        item["expected_doc"] = chosen["doc"]
        item["expected_page"] = chosen["page"]
        confirmed += 1

    EVAL_SET.write_text(json.dumps(items, indent=2) + "\n")
    print(f"\n{confirmed}/{len(items)} questions have a confirmed answer key -> {EVAL_SET}")


if __name__ == "__main__":
    main()
