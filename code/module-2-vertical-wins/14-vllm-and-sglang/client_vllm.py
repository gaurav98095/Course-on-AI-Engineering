"""Point the RAG pipeline's generation step at a real serving engine
instead of loading the model in-process -- exactly the API-layer /
GPU-layer split Lecture 02 first named. Retrieval is unchanged, local,
ours; generation is now a call to vLLM or SGLang.

Both engines speak the OpenAI-compatible chat completions API, so this
one script works against either -- just point --port at whichever one
you started (vLLM defaults to 8000, SGLang to 30000).

Run:  vllm serve Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 \
        --max-model-len 16384 --limit-mm-per-prompt '{"image":2,"video":0}'   # terminal 1
      python client_vllm.py "Why does an aircraft stall at the critical angle of attack?"   # terminal 2
      python client_vllm.py "..." --port 30000   # against SGLang instead
"""

import argparse
import base64

from openai import OpenAI

from rag import CORPUS, GENERATOR, SYSTEM, Retriever


def build_messages(question: str, hits_t: list[dict], hits_i: list[dict]) -> list[dict]:
    content = []
    for m in hits_i:   # retrieved figures travel as base64 images, same idea as Lecture 02's client.py
        b64 = base64.b64encode((CORPUS / "images" / m["file"]).read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    excerpts = "\n\n".join(f"[{t['doc']} p.{t['page']}] {t['text']}" for t in hits_t)
    content.append({"type": "text", "text": f"Manual excerpts:\n{excerpts}\n\nQuestion: {question}"})
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--port", type=int, default=8000, help="8000 for vLLM, 30000 for SGLang")
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{args.port}/v1")

    retriever = Retriever()   # unchanged from Lecture 01 -- retrieval never moved
    hits_t, hits_i = retriever(args.question)
    messages = build_messages(args.question, hits_t, hits_i)

    response = client.chat.completions.create(
        model=GENERATOR, messages=messages, max_tokens=args.max_tokens,
    )

    print("\n--- retrieved:")
    for t in hits_t:
        print(f"  [{t['doc']} p.{t['page']}] {t['text'][:80]}...")
    print(f"\n--- answer:\n{response.choices[0].message.content}")


if __name__ == "__main__":
    main()
