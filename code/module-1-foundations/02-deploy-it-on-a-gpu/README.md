# Lecture 02 — Deploy It on a GPU

Self-contained: this folder is a **copy-forward** of `01b-gpu-vitals/` plus an HTTP endpoint. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 02 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/02-deploy-it-on-a-gpu.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| 💻 Your laptop | Browser: exposing the Studio's port; optionally calling the public URL |
| ⚡ Lightning AI Studio | Runs the server. Unlabeled commands run in its terminal. |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Lecture 01b

| File | What it is |
| --- | --- |
| `serve.py` | LitServe HTTP endpoint — model loads once in `setup`; adds a `/health` route and a `/metrics` route (API-layer counters + live GPU vitals via `pynvml`) |
| `client.py` | Calls the endpoint from anywhere, prints timing breakdown |

(`ingest.py`, `rag.py`, `measure.py`, `gpu_vitals.py`, `plot_vitals.py`, and `data/` are copied forward from Lecture 01b unchanged, so this folder still runs on its own.)

**Why `/metrics` reads a log file instead of an object's attributes**: LitServe runs `predict` in a separate worker process from the one serving HTTP routes, so a Python object's attributes set inside `predict` are invisible to a route registered on `server.app`. `serve.py` has the worker append one JSON line per request to `metrics.jsonl` (gitignored, regenerated every run); the `/metrics` route reads that file back. GPU vitals don't need this — `pynvml` talks to the driver directly and works from either process. See the lecture's Step 6 for the full story.

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-1-foundations/02-deploy-it-on-a-gpu
pip install -r requirements.txt
python ingest.py                 # build the indexes (skip if you already have corpus/)
```

⚡ Terminal 1 — start the server:

```bash
python serve.py
```

```text
models loaded, serving            # after ~90 s cold start
INFO: Uvicorn running on http://0.0.0.0:8000
```

⚡ Terminal 2 — health check, a real call, and the metrics route:

```bash
curl localhost:8000/health
python client.py "Why does an aircraft stall at the critical angle of attack?"
python client.py "What does this instrument do?" --image data/sample-query-instrument.jpg
curl localhost:8000/metrics
```

💻 On your laptop — expose port 8000 from the Studio's port plugin, then call the public URL from anywhere:

```bash
python client.py "How does the altimeter work?" --url https://<your-exposed-url>
```

## Troubleshooting

- **`/health` hangs**: the server is still loading the model — wait for "models loaded, serving".
- **Requests time out**: `serve.py` sets `timeout=120`; long answers or a cold GPU can exceed it — see Lecture 03 for why.
- **`AttributeError: 'LitServer' object has no attribute 'app'`**: your installed LitServe version doesn't expose the FastAPI instance the same way. Check `pip show litserve` and consult its docs for the current attribute name; alternatively, run a second tiny script with its own `FastAPI()`/`uvicorn.run()` on a different port (e.g. 8001) that reads `metrics.jsonl` the same way `metrics_route()` does — the file-based handoff still works regardless of which process reads the file.
- **`/metrics` shows all zeros right after starting the server**: `metrics.jsonl` doesn't exist until the first request completes — that's expected, not a bug.
- Everything from Lecture 01/01b's troubleshooting applies here too.

---

Previous: [`../01b-gpu-vitals/`](../01b-gpu-vitals/) · Next: [`../03-load-test-it-until-it-breaks/`](../03-load-test-it-until-it-breaks/) — the same endpoint, under siege.
