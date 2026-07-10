"""Deploy: wrap the RAG in an HTTP endpoint with LitServe.

The model loads ONCE (in setup), then answers requests forever.

Run:   python serve.py                   # -> http://localhost:8000, 1 worker (Lecture 03's ceiling)
       python serve.py --workers 2       # a second till on the same GPU (Lecture 03b)
Test:  curl localhost:8000/health
       curl localhost:8000/metrics
       python client.py "Why does an aircraft stall?"
"""

import argparse
import base64
import io
import json
import time
from pathlib import Path

import litserve as ls
import pynvml
from PIL import Image

from rag import Generator, Retriever

# `predict` and this route are NOT the same OS process (see "One Process,
# Two Systems" below) -- LitServe runs inference in a forked worker process,
# separate from the FastAPI process our /metrics route lives in. Python
# objects don't cross that boundary. A small append-only log file does.
METRICS_LOG = Path("metrics.jsonl")


class RAGAPI(ls.LitAPI):
    def setup(self, device):
        """Runs exactly once, when the WORKER process starts.

        This is the cold start: ~90 s of model loading on an L40S. It runs
        once per worker, in the worker's own process -- not in the process
        that will answer /health or /metrics.
        """
        self.retriever = Retriever()
        self.generator = Generator()
        METRICS_LOG.touch()
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
        """The actual work: retrieve, generate, time it. Runs in the worker."""
        question, image, max_new = inputs
        t0 = time.perf_counter()
        ok = True
        n_in = n_out = 0
        try:
            hits_t, hits_i = self.retriever(question, image)
            t_retrieve = time.perf_counter() - t0

            answer, n_in, n_out = self.generator(
                question, hits_t, hits_i, image, max_new_tokens=max_new
            )
            t_total = time.perf_counter() - t0
        except Exception:
            ok = False
            t_total = time.perf_counter() - t0
            raise
        finally:
            # the ONLY thing that crosses the process boundary: one JSON
            # line, appended by the worker, read later by the API process
            with open(METRICS_LOG, "a") as f:
                f.write(json.dumps({
                    "t": time.time(), "seconds": round(t_total, 4),
                    "new_tokens": n_out, "ok": ok,
                }) + "\n")

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


_nvml_ready = False


def gpu_handle():
    """pynvml talks to the driver, not to a CUDA context -- unlike the
    worker's counters, GPU vitals are readable from ANY process, no log
    file needed for this half of /metrics.
    """
    global _nvml_ready
    if not _nvml_ready:
        pynvml.nvmlInit()
        _nvml_ready = True
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def metrics_route():
    """Runs in the API/FastAPI process. Reads the worker's log file for
    API-layer counters, queries the driver directly for GPU-layer vitals.
    """
    if METRICS_LOG.exists():
        records = [json.loads(line) for line in METRICS_LOG.read_text().splitlines() if line]
    else:
        records = []
    recent = records[-100:]   # bound the latency window; totals stay exact

    latencies = sorted(r["seconds"] for r in recent if r["ok"])
    p50 = latencies[len(latencies) // 2] if latencies else 0.0

    handle = gpu_handle()
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return {
        # API-layer metrics: what a load balancer / autoscaler would read
        "requests_total": len(records),
        "errors_total": sum(1 for r in records if not r["ok"]),
        "tokens_generated_total": sum(r["new_tokens"] for r in records),
        "latency_p50_s": round(p50, 2),
        # real, not a placeholder: LitServe's own dispatch queue, summed
        # across every worker process, via track_requests=True below
        # (Lecture 03b)
        "in_flight_estimate": server.active_requests or 0,
        # GPU-layer metrics: what a GPU fleet dashboard would read
        "gpu_util_pct": util.gpu,
        "gpu_mem_used_mib": round(mem.used / 2**20),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1,
                    help="worker processes on this one GPU (Lecture 03b)")
    args = ap.parse_args()

    api = RAGAPI()
    # workers_per_device=1 was Lecture 03's deliberately naive default -- one
    # till, one line. track_requests=True turns on LitServe's own in-flight
    # counter, summed across however many workers we start below (Lecture 03b).
    server = ls.LitServer(api, accelerator="gpu", timeout=120,
                           workers_per_device=args.workers, track_requests=True)

    # LitServe is built on FastAPI; .app is the underlying FastAPI instance.
    # (If your installed LitServe version doesn't expose .app, see this
    # lecture's README troubleshooting section.)
    server.app.get("/metrics")(metrics_route)

    print(f"serving with {args.workers} worker(s) on this GPU")
    server.run(port=8000)
