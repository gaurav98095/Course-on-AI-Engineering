# Lecture 07 — Quantization I: Number Formats

Self-contained: this folder is a **copy-forward** of Module 1's final lecture folder (`06-profiling-where-the-time-actually-goes/`) plus two new scripts. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 07 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/07-quantization-i-number-formats.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| ⚡ Lightning AI Studio | Everything — `quantize_and_measure.py` needs a GPU; `bit_formats.py` doesn't but is meant to run alongside it |
| 💻 Your laptop | Browser only, reading the lecture |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Module 1's final lecture

| File | What it is |
| --- | --- |
| `bit_formats.py` | No GPU needed — rounds real numbers through fp16/bf16, shows the raw bits, and demonstrates fp16's overflow at 70000.0 |
| `quantize_and_measure.py` | Loads the course model at bf16 / int8 / int4 via `bitsandbytes` and measures real weight memory + decode speed for each |

(Everything from Module 1 — `ingest.py`, `rag.py`, `serve.py` with its `/metrics` route, `gpu_vitals.py`, `roofline.py`, `kv_cache.py`, `profile_request.py`, and `data/` — is copied forward unchanged, so this folder still runs the full Module 1 system on its own.)

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-2-vertical-wins/07-quantization-i-number-formats
pip install -r requirements.txt     # adds bitsandbytes
```

```bash
python bit_formats.py
```

Then, one mode at a time (each loads a fresh 8B-parameter model — don't run two at once on a single GPU):

```bash
python quantize_and_measure.py --mode bf16
python quantize_and_measure.py --mode int8
python quantize_and_measure.py --mode int4
```

## Troubleshooting

- **`bitsandbytes` import/CUDA errors**: it ships prebuilt kernels for specific CUDA versions — if import fails, check `pip show bitsandbytes` against your Studio's CUDA version, or reinstall with `pip install -U bitsandbytes`.
- **Quantized load errors on the vision tower**: pass `llm_int8_skip_modules=["visual"]` (or the correct submodule name — check `model.named_modules()` on a bf16 load first) inside `BitsAndBytesConfig` to leave the vision tower unquantized while still shrinking the language model.
- **int8 or int4 run is *slower* than bf16**: this isn't a bug — it's the lecture's actual point. See "Where It Breaks" in the lecture and Math Deep Dive 07.
- Everything from Module 1's troubleshooting (see `../../module-1-foundations/*/README.md`) still applies if you also run that system in this folder.

---

Previous: [`../../module-1-foundations/06-profiling-where-the-time-actually-goes/`](../../module-1-foundations/06-profiling-where-the-time-actually-goes/) · This is the first lecture in Module 2 — more lectures land here as the course continues.
