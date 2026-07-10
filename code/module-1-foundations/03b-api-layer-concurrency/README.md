# Lecture 03b — Fix It at the API Layer, First

Self-contained: this folder is a **copy-forward** of `03-load-test-it-until-it-breaks/` with one file changed — `serve.py` now takes a `--workers` flag and reports a real `in_flight_estimate`. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 03b on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/03b-api-layer-concurrency.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| ⚡ Lightning Studio, terminal 1 | `serve.py --workers N` — the same victim, now with N tills |
| ⚡ Lightning Studio, terminal 2 | `load_test.py` — the identical swarm from Lecture 03, unchanged |
| ⚡ Lightning Studio, terminal 3 | `live_dashboard.py` — now shows a real `in_flight_estimate`, not a placeholder zero |
| 💻 Your laptop | Optional: rerun the swarm over the public URL and compare |

## What's new versus Lecture 03

| File | What changed |
| --- | --- |
| `serve.py` | Adds `--workers` (maps to LitServe's `workers_per_device`) and `track_requests=True`, so `/metrics`'s `in_flight_estimate` reads LitServe's real dispatch-queue counter instead of a hardcoded `0` |

Everything else — `load_test.py`, `live_dashboard.py`, `gpu_vitals.py`, `plot_vitals.py`, `rag.py`, `ingest.py` — is copied forward **unchanged**. Same swarm, same harness, only the server's worker count changes. `diff serve.py ../03-load-test-it-until-it-breaks/serve.py` shows the entire lecture.

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-1-foundations/03b-api-layer-concurrency
pip install -r requirements.txt
python ingest.py                 # build the indexes (skip if you already have corpus/)
```

⚡ Terminal 1 — baseline, one till (same as Lecture 03):

```bash
python serve.py --workers 1
```

⚡ Terminal 2 — the identical sweep:

```bash
python load_test.py --sweep 1 2 4 8 16 32
```

Now open a second till. Stop `serve.py`, restart it:

⚡ Terminal 1:

```bash
python serve.py --workers 2
```

⚡ Terminal 2 — rerun the exact same sweep, compare the table:

```bash
python load_test.py --sweep 1 2 4 8 16 32
```

⚡ Terminal 3 — watch the queue drain across both workers live:

```bash
python live_dashboard.py
```

Then push it until the stockroom runs out:

```bash
python serve.py --workers 3     # watch nvidia-smi / gpu_vitals.py -- does this OOM on your GPU?
```

## Troubleshooting

- **`--workers 2` doubles the cold start wait**: expected — each worker calls `setup()` independently and loads its own copy of the model. Wait for both `models loaded, serving` lines before load-testing.
- **`--workers 3` (or higher) OOMs**: that's the lecture's punchline, not a bug — see "Where It Breaks" on the course page. Lower `--workers` or use a GPU with more VRAM to push the ceiling further.
- **`in_flight_estimate` stays 0 even under load**: confirm `track_requests=True` is present in your `serve.py` and that you're hitting `/metrics` on the same server process you're load-testing.
- Everything from Lectures 01/01b/02/03's troubleshooting applies here too.

---

Previous: [`../03-load-test-it-until-it-breaks/`](../03-load-test-it-until-it-breaks/) · Next: [`../04-the-gpu-architecture-and-roofline/`](../04-the-gpu-architecture-and-roofline/) — explaining the <1% per-request GPU utilization number that replicas alone can't fix.
