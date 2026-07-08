"""Deploy: wrap the RAG in an HTTP endpoint with LitServe.

The model loads ONCE (in setup), then answers requests forever.

Run:   python serve.py                 # -> http://localhost:8000
Test:  curl localhost:8000/health
       curl localhost:8000/metrics
       python client.py "Why does an aircraft stall?"
"""

import base64
import io
import time

import litserve as ls
import pynvml
from PIL import Image

from rag import Generator, Retriever


class RAGAPI(ls.LitAPI):
    def setup(self, device):
        """Runs exactly once, when the server starts — never per request.

        This is the cold start: ~90 s of model loading on an L40S.
        """
        self.retriever = Retriever()
        self.generator = Generator()

        # API-layer counters — this is the "state" our /metrics route reads
        self.request_count = 0
        self.error_count = 0
        self.total_new_tokens = 0
        self.latencies_s = []          # last 100 request times, for a crude p50

        # GPU vitals — same pynvml calls as Lecture 01b's gpu_vitals.py
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

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
        try:
            hits_t, hits_i = self.retriever(question, image)
            t_retrieve = time.perf_counter() - t0

            answer, n_in, n_out = self.generator(
                question, hits_t, hits_i, image, max_new_tokens=max_new
            )
            t_total = time.perf_counter() - t0
        except Exception:
            self.error_count += 1
            raise

        # API-layer bookkeeping: what /metrics will report
        self.request_count += 1
        self.total_new_tokens += n_out
        self.latencies_s.append(t_total)
        self.latencies_s = self.latencies_s[-100:]

        # one log line per request: still useful for a human watching live
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


def make_metrics_route(api: RAGAPI):
    """A closure so the route can read the live RAGAPI instance's state."""
    def metrics():
        lat = sorted(api.latencies_s)
        p50 = lat[len(lat) // 2] if lat else 0.0

        util = pynvml.nvmlDeviceGetUtilizationRates(api.gpu_handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(api.gpu_handle)

        return {
            # API-layer metrics: what a load balancer / autoscaler would read
            "requests_total": api.request_count,
            "errors_total": api.error_count,
            "tokens_generated_total": api.total_new_tokens,
            "latency_p50_s": round(p50, 2),
            "in_flight_estimate": 0,          # honest today: one worker, no queue depth to report yet
            # GPU-layer metrics: what a GPU fleet dashboard would read
            "gpu_util_pct": util.gpu,
            "gpu_mem_used_mib": round(mem.used / 2**20),
        }
    return metrics


if __name__ == "__main__":
    api = RAGAPI()
    # one worker, one GPU, no batching — deliberately naive; Lecture 03 breaks it
    server = ls.LitServer(api, accelerator="gpu", timeout=120)

    # LitServe is built on FastAPI; .app is the underlying FastAPI instance.
    # (If your installed LitServe version doesn't expose .app, see this
    # lecture's README troubleshooting section.)
    server.app.get("/metrics")(make_metrics_route(api))

    server.run(port=8000)
