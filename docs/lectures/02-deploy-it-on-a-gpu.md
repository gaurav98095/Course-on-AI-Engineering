---
layout: default
title: "Lecture 02 — Deploy It on a GPU"
---

# Lecture 02 — Deploy It on a GPU

> **In one sentence:** We turn last lecture's script into a running service — an HTTP endpoint with a schema, a health check, per-request logs, and a metrics route — and catch it quietly conflating two systems (an API layer and a GPU layer) that production keeps apart.

## Learning Objectives

- Wrap the multimodal RAG in a LitServe endpoint where the model loads once and answers forever.
- Trace the anatomy of one request — network, queue, retrieve, prefill, decode — and know which part you're measuring.
- Expose the endpoint to the internet from a Lightning Studio, call it like a user, and read your first request logs.
- Add a real `/metrics` endpoint — and hit a real bug doing it, one that exposes exactly how the API layer and GPU layer are already separated under the hood.

## Prerequisites

| Concept | Needed? | Notes |
| --- | --- | --- |
| Lecture 01 | Yes | The RAG must be built and `corpus/` indexed on your Studio |
| Lecture 01b | Light | We reuse its GPU-vitals code inside today's `/metrics` endpoint |
| HTTP basics | Light | You've seen a POST request and JSON before |
| GPUs | No | Still renting; internals arrive in Lecture 04 |

## Story

Your Lecture 01 demo went well. Too well.

Your manager watched the terminal print a cited answer, nodded, and said the seven words that define this lecture: *"Great — can you send me a link?"*

A script that runs when **you** type is not a product. A **service** answers whoever calls, whenever they call, without you in the room. That gap — script to service — is where most ML projects stall.

A century ago, the telephone network had the same problem: a phone on your desk is useless until something always-on connects calls to answers.

<figure>
  <img src="../assets/images/switchboard.jpg" alt="Telephone operators seated at a large switchboard, early 1900s">
  <figcaption>A telephone switchboard, early 1900s: always on, one operator per call, every call logged. Today we build exactly this for our model. <em>Photo: Wikimedia Commons, public domain</em></figcaption>
</figure>

## Mental Model

> **A deployed model is a switchboard.** There's a number to call (the endpoint), an operator who answers (the server process), a fixed way to state your business (the request schema), and a logbook (per-request logs). And — crucially, today — **one operator**: while she connects your call, everyone else hears ringing.

Map it once and the code becomes obvious:

| Switchboard | Our service |
| --- | --- |
| The phone number | `http://<host>:8000/predict` |
| The operator | the LitServe worker process |
| "State your business" | the JSON schema in `decode_request` |
| Connecting the call | `predict` — retrieve, then generate |
| The logbook | one printed line per request |
| Operator's training, done before opening | `setup` — the model loads **once** |

One operator per switchboard is exactly one worker per GPU. Remember the ringing — Lecture 03 makes it deafening.
{: .remember}

## The System

Where things run today, one new column compared to Lecture 01: the **caller** can now be anywhere.

| Environment | Role in this lecture |
| --- | --- |
| 💻 Your laptop | Browser: reading, clicking "expose port" in Lightning; optionally running `client.py` against the public URL |
| ⚡ Lightning Studio | Runs the server; unlabeled commands run in its terminal |
| ☁️ AWS | Still nothing — Module 3 |

One request, end to end:

```mermaid
sequenceDiagram
    participant C as Client (anywhere)
    participant S as LitServe (Studio, port 8000)
    participant G as GPU
    C->>S: POST /predict {question, image_b64?}
    S->>S: decode_request — JSON to objects
    S->>G: retrieve (FAISS + encoders)
    S->>G: generate (prefill, then decode)
    G-->>S: answer, token counts
    S-->>C: encode_response — JSON back
```

Everything from Lecture 01 is unchanged underneath — `serve.py` just picks up `Retriever` and `Generator` from `rag.py` and puts a phone number on them.

## The Build

This lecture's folder, `code/module-1-foundations/02-deploy-it-on-a-gpu/`, is a copy-forward of Lecture 01b's folder (so `gpu_vitals.py` rides along) with two new files: `serve.py` and `client.py`.

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git   # skip if already cloned
cd Course-on-AI-Engineering/code/module-1-foundations/02-deploy-it-on-a-gpu
pip install -r requirements.txt     # adds litserve
python ingest.py                    # rebuild the indexes in this folder
```

### Step 1 — Write the endpoint

The whole server is one class with four methods, and the method boundaries *are* the lesson:

```python
class RAGAPI(ls.LitAPI):
    def setup(self, device):            # runs ONCE at startup
        self.retriever = Retriever()
        self.generator = Generator()

    def decode_request(self, request):  # JSON in -> python objects
        question = request["question"]
        image = ...                     # optional base64 image
        return question, image, max_new

    def predict(self, inputs):          # the actual work, timed and logged
        hits_t, hits_i = self.retriever(question, image)
        answer, n_in, n_out = self.generator(question, hits_t, hits_i, image)
        return {"answer": answer, "sources": [...], "seconds": ...}

    def encode_response(self, output):  # python objects -> JSON out
        return output
```

The line that separates professionals from demos is the first one: **the model loads in `setup`, never in `predict`.** Load-per-request would add ~90 seconds to every single answer.

### Step 2 — Start it

```bash
python serve.py
```

What you should see (ballpark — the wait is Qwen3-VL loading into VRAM):

```text
models loaded, serving            # after ~90 s
INFO: Uvicorn running on http://0.0.0.0:8000
```

That ~90 seconds has a name — **cold start** — and it's why services restart rarely and autoscaling (Module 3) is harder than it sounds.

### Step 3 — Health check, then a real call

⚡ *In a second Studio terminal* (the first one is busy being the server):

```bash
curl localhost:8000/health          # -> ok  (LitServe gives you this for free)
python client.py "Why does an aircraft stall at the critical angle of attack?"
```

The client prints the answer, its sources, and a line worth staring at:

```text
server time: 8.9s | wall time incl. network: 8.94s | overhead: 0.04s
```

An image query works over HTTP too — the photo travels as base64 text inside the JSON:

```bash
python client.py "What does this instrument do?" --image data/sample-query-instrument.jpg
```

### Step 4 — Give your manager the link

💻 *On your laptop, in the browser:* in the Studio's right-hand toolbar, open the **port** plugin and expose port **8000**. Lightning gives you a public `https://...` URL.

From any machine on earth:

```bash
python client.py "How does the altimeter work?" --url https://<your-url>
```

<figure class="portrait">
  <img src="../assets/images/telephone-caller.jpg" alt="Man in a Stetson hat talking on a candlestick telephone">
  <figcaption>The whole point of a switchboard: the caller can be anyone, anywhere — they just need the number. <em>Photo: Sam Hood, public domain</em></figcaption>
</figure>

That URL is the deliverable your manager asked for. It is also — say it out loud — **an open door to a GPU you pay for**, with no authentication. Exercise 1 fixes that; Module 3 does it properly.

### Step 5 — Read your logs

Back in the server terminal, every request left one line:

```text
[req] retrieve=41ms total=8.92s prompt_tok=1847 new_tok=243 tok/s=27.2
```

One line, four numbers, and the whole story of this course is already in them: retrieval is **milliseconds**, generation is **seconds** — the GPU spends its life in `generate`. This log line is observability v0; Prometheus dashboards (Lecture 27) are this same line, grown up.

### Step 6 — Expose real metrics, not just logs

A log line is for a human, watching one line at a time. A **metrics endpoint** is for a machine — a dashboard, an autoscaler, an alert — to *scrape* on a schedule and aggregate. The obvious first design: track a few counters as attributes on `RAGAPI`, set them up in `setup`, update them in `predict`, and read them straight out of the object in the route:

```python
def setup(self, device):
    self.request_count = 0
    ...

def predict(self, inputs):
    ...
    self.request_count += 1     # looks fine. it is not fine.
```

**This design is broken, and it's worth seeing exactly why**, because the reason is the whole point of this lecture. LitServe doesn't run `predict` in the same process that answers HTTP requests — it forks a separate **worker process** to run inference, so the GPU work never blocks the web server. `setup` and `predict` execute inside that worker. A route registered on `server.app`, like our `/metrics`, executes in the *original* process. They are not the same Python interpreter, do not share memory, and an attribute set on `self` inside the worker is invisible to code running anywhere else — the route would either crash reading `api.gpu_handle` (never set in this process) or silently read a `request_count` that never moves.

We didn't get to choose whether our API layer and our GPU layer are separate systems. **LitServe already made that choice for us.** The fix has to cross a real process boundary, so it uses the simplest thing that can: the worker appends one JSON line per request to a small log file; the route reads that file back.

```python
# in predict() -- runs in the WORKER process
with open(METRICS_LOG, "a") as f:
    f.write(json.dumps({"seconds": t_total, "new_tokens": n_out, "ok": ok}) + "\n")

# in metrics_route() -- runs in the API process, reads what the worker wrote
records = [json.loads(line) for line in METRICS_LOG.read_text().splitlines()]
```

GPU vitals don't have this problem at all — `pynvml` talks directly to the NVIDIA driver, not to a specific process's CUDA context, so `metrics_route()` can query utilization and memory itself, no file needed:

```python
def metrics_route():
    records = [json.loads(l) for l in METRICS_LOG.read_text().splitlines()]
    handle = gpu_handle()                          # this process's own pynvml call — always safe
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return {
        "requests_total": len(records), "errors_total": sum(1 for r in records if not r["ok"]),
        "gpu_util_pct": util.gpu, ...
    }
```

```bash
curl localhost:8000/metrics
```

```json
{"requests_total": 4, "errors_total": 0, "tokens_generated_total": 812,
 "latency_p50_s": 8.7, "in_flight_estimate": 0,
 "gpu_util_pct": 12, "gpu_mem_used_mib": 17240}
```

> If your installed LitServe version doesn't expose `server.app`, check the folder's README — the fallback is a second tiny FastAPI/Uvicorn process reading the same log file, which is uglier but always works.

## Measure It

The same baseline questions as Lecture 01, but now asked through HTTP instead of a direct script call. Ballpark on one L40S, bf16, batch 1:

| Metric | Lecture 01 (direct) | Lecture 02 (HTTP) | What it says |
| --- | --- | --- | --- |
| Mean answer time | ~8–10 s | ~8–10 s + ~0.05 s | Network overhead is noise |
| Retrieval share | ~40 ms | ~40 ms | Still not the problem |
| Cold start | — | ~90 s | The cost of every restart |
| Concurrent callers | 1 (by definition) | **still 1** | The switchboard has one operator |

> Deploying changed *who can call* — it changed nothing about *how much can be served*. HTTP added ~0.5% overhead; the other 99.5% is still the model.

## One Process, Two Systems

Look again at the `/metrics` output from Step 6. It's answering two completely different questions in one JSON blob:

> **`requests_total`, `latency_p50_s`, `errors_total`** — these describe the **API layer**: is the *service* healthy? Would a load balancer route traffic here? This is CPU-cheap bookkeeping that has nothing to do with matrix multiplication.
>
> **`gpu_util_pct`, `gpu_mem_used_mib`** — these describe the **model-serving layer**: is the *GPU* healthy? Is it about to OOM? This is expensive, hardware-bound state that has nothing to do with HTTP.

Step 6 already forced a small taste of this: LitServe itself splits `serve.py` into two OS processes (a web-facing one, a GPU-facing worker) just to keep HTTP responsive while the GPU is busy — that's *why* `/metrics` needed a log file instead of a shared Python object. But `serve.py` is still **one deployable unit** — one command, one machine, one GPU, both processes living and dying together. That was the right simplification for Lectures 01–02: one file, one mental model, nothing to configure. It quietly hides a mismatch that gets expensive to ignore at scale.

| | API layer | Model-serving layer |
| --- | --- | --- |
| Bottleneck | Network, CPU, JSON parsing | GPU compute and memory (Lecture 04) |
| Cost to run | Cents — any small CPU instance | Dollars/hour — a whole GPU, idle or not |
| How it scales | Horizontally, cheaply — spin up 50 more pods | Vertically and expensively — each replica needs its own GPU |
| Failure mode | Slow JSON parsing, connection floods | OOM, thermal throttling, cold starts (~90 s, Step 2) |
| Restart cost | Milliseconds | The ~90 second cold start you just watched |

A production system almost never puts both in one process. A **gateway** (API layer) — stateless, cheap, trivially replicated — sits in front of a small pool of **model workers** (GPU layer), each one expensive and precious, routed to by the gateway. Our `RAGAPI.predict` doing both jobs at once is a teaching simplification, not a production pattern — and it's exactly what breaks first when Lecture 03 puts this system under real load.

> This split has a name and a payoff: **prefill/decode disaggregation** (Lecture 24) takes the same idea one step further and splits the *GPU* layer itself in two. **Kubernetes** (Lecture 26) is where the gateway/worker split becomes real infrastructure, with each side autoscaling on its own metrics — the API layer on request rate, the GPU layer on the very `gpu_util_pct` number Step 6 just started exposing.

Conflating two systems with different bottlenecks, different costs, and different failure modes into one process is a fine way to learn. It's a poor way to run one at scale — remember this table; Module 3 is largely about correcting it.
{: .remember}

## The Math, One Level Deeper

No new derivation today — just the equation that names the parts, because from now on every latency number we measure decomposes as:

\\[
T_{\text{request}} \;=\; T_{\text{network}} \;+\; T_{\text{queue}} \;+\; T_{\text{retrieve}} \;+\; T_{\text{prefill}} \;+\; T_{\text{decode}}
\\]

Today we measured: network ≈ 0.05 s, queue = 0 (one polite caller), retrieve ≈ 0.04 s, prefill + decode ≈ 9 s. Every optimization in this course attacks exactly one term — and next lecture, \\(T_{\text{queue}}\\) stops being zero and tries to eat the whole equation. That story has beautiful math (Little's law, why p95 explodes), and it gets its own deep-dive page next time.

## Where It Breaks

**One operator.** A second caller waits for the first to finish — completely, all ~9 seconds. There is no parallelism anywhere in `serve.py`. This is deliberate: you must *feel* it break in Lecture 03 before Module 2's fixes mean anything.

**The open door.** The exposed URL has no auth, no rate limit. Anyone with the link can spend your GPU credits.

**Base64 bloat.** Encoding images as text inflates them ~33%, and a phone photo can be 5 MB. Fine today; at scale you'll want size limits and object storage, not JSON stuffing.

**Timeouts are policy.** We set `timeout=120`. Too low and long answers die mid-generation; too high and a stuck request holds the only operator hostage. There is no correct value — only a chosen one.

**`metrics.jsonl` grows forever and isn't safe for concurrent writers.** One worker appending one line at a time is fine; Lecture 03's multiple simultaneous requests are still handled one at a time by our single worker, so writes never interleave — but this file-based approach is a teaching-scale stopgap, not what a real fleet does (Lecture 27's Prometheus setup replaces it properly, and neither approach would survive multiple GPU workers writing to the same file without real care).

A service's weakest points are rarely the model: they're the door (auth), the line (queue), the clock (timeouts), and — as Step 6 just proved the hard way — the wall between one process and another.
{: .remember}

## Exercises

1. **Lock the door.** Add an `X-API-Key` header check in `decode_request` (reject with an exception if it's wrong). Confirm a bad key fails and a good key works.
2. **Schema power.** The endpoint already accepts `max_new_tokens`. Call with 50 vs 400 and compare `seconds` — you just built a cost knob into your API.
3. **Measure the network honestly.** Run `client.py` from your laptop against the public URL and compare `overhead` with the in-Studio value. Where did the extra go?
4. **Two callers, one operator.** Open two terminals, fire `client.py` in both simultaneously, and note both wall times. Write the second number down — Lecture 03 explains it exactly.
5. **Watch `/metrics` move.** Fire five requests, `curl localhost:8000/metrics` after each one, and confirm `requests_total` and `tokens_generated_total` climb correctly. Now break something on purpose (send a malformed request) and confirm `errors_total` moves too.
6. **Draw the split.** On paper, sketch today's one-process `serve.py` as two boxes instead of one — a gateway box and a GPU-worker box — and label which of `/metrics`' seven fields belongs on each side of the line. You just designed the shape Lecture 26 builds for real.

## Summary

We put a phone number on the model: `setup` loads once, `decode_request` defines the schema, `predict` does timed work, `encode_response` answers in JSON. A health check says "alive", a public URL makes it real, a log line tells a human the truth, and a `/metrics` route tells a machine the same truth in a form it can scrape — and building that route the naive way actually broke, because LitServe already runs the GPU work in a separate process from the one answering HTTP. One deployment, two roles, and — it turns out — two processes underneath already, just not yet two independently-scaled, independently-costed *systems*, which is what production architectures deliberately build toward.

> **What should you remember?**
> - Load in `setup`, never in `predict` — cold start is paid once, not per request.
> - \\(T_{\text{request}} = T_{\text{net}} + T_{\text{queue}} + T_{\text{retrieve}} + T_{\text{prefill}} + T_{\text{decode}}\\): know which term you're measuring.
> - The API layer and the GPU layer have different bottlenecks, costs, and failure modes — one process today, two systems by Module 3.

## Resources

- LitServe documentation — `lightning.ai/docs/litserve` (the `LitAPI` lifecycle we used).
- The Twelve-Factor App — factor VI (processes) and IX (disposability) are `setup` and cold starts in older clothes.
- Lightning AI docs: exposing Studio ports.
- The Prometheus exposition format — the de facto standard `/metrics` shape our JSON version is a simplified preview of; Lecture 27 uses it properly.

---

[← Previous: Lecture 01b — GPU Vitals: Watching What You Built](01b-gpu-vitals.md) · [Course Home](../index.md) · [Next: Lecture 03 — Load-Test It Until It Breaks →](03-load-test-it-until-it-breaks.md)
