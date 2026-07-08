"""Deploy: wrap the RAG in an HTTP endpoint with LitServe.

The model loads ONCE (in setup), then answers requests forever.

Run:   python serve.py                 # -> http://localhost:8000
Test:  curl localhost:8000/health
       python client.py "Why does an aircraft stall?"
"""

import base64
import io
import time

import litserve as ls
from PIL import Image

from rag import Generator, Retriever


class RAGAPI(ls.LitAPI):
    def setup(self, device):
        """Runs exactly once, when the server starts — never per request.

        This is the cold start: ~90 s of model loading on an L40S.
        """
        self.retriever = Retriever()
        self.generator = Generator()
        print("models loaded, serving")

    def decode_request(self, request):
        """JSON in -> python objects. The API schema lives here."""
        question = request["question"]
        image = None
        if request.get("image_b64"):
            raw = base64.b64decode(request["image_b64"])
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        max_new = int(request.get("max_new_tokens", 400))
        return question, image, max_new

    def predict(self, inputs):
        """The actual work: retrieve, generate, time it."""
        question, image, max_new = inputs
        t0 = time.perf_counter()
        hits_t, hits_i = self.retriever(question, image)
        t_retrieve = time.perf_counter() - t0

        answer, n_in, n_out = self.generator(
            question, hits_t, hits_i, image, max_new_tokens=max_new
        )
        t_total = time.perf_counter() - t0

        # one log line per request: this is your observability, v0
        print(f"[req] retrieve={t_retrieve*1000:.0f}ms total={t_total:.2f}s "
              f"prompt_tok={n_in} new_tok={n_out} tok/s={n_out/t_total:.1f}")

        return {
            "answer": answer,
            "sources": [f"{t['doc']} p.{t['page']}" for t in hits_t]
                       + [m["file"] for m in hits_i],
            "prompt_tokens": n_in,
            "new_tokens": n_out,
            "seconds": round(t_total, 2),
        }

    def encode_response(self, output):
        """python objects -> JSON out. Already JSON-shaped here."""
        return output


if __name__ == "__main__":
    api = RAGAPI()
    # one worker, one GPU, no batching — deliberately naive; Lecture 03 breaks it
    server = ls.LitServer(api, accelerator="gpu", timeout=120)
    server.run(port=8000)
