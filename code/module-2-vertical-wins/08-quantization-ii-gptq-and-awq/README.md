# Lecture 08 — Quantization II: GPTQ & AWQ in Practice

Self-contained: this folder is a **copy-forward** of `07-quantization-i-number-formats/` plus two new scripts. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 08 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/08-quantization-ii-gptq-and-awq.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| ⚡ Lightning AI Studio | Everything — quantization and benchmarking both need a GPU |
| 💻 Your laptop | Browser only, reading the lecture |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Lecture 07

| File | What it is |
| --- | --- |
| `quantize_gptq_awq.py` | Quantizes the course model with GPTQ or AWQ (`--method gptq`/`--method awq`), calibrated on our own Lecture 01 corpus, via `GPTQModel` |
| `benchmark_quantized.py` | Loads a saved quantized model, measures real decode speed and memory, and re-asks Lecture 01's question as a quality spot check |

(Everything from Lectures 01–07 — including `bit_formats.py` and `quantize_and_measure.py` from Lecture 07 — is copied forward unchanged, so this folder still runs the full system on its own.)

**Note on the library**: we use [`GPTQModel`](https://github.com/ModelCloud/GPTQModel), the current actively-maintained library for this. The older standalone `AutoGPTQ` package is no longer the integration Hugging Face `transformers` recommends — `GPTQModel` covers both GPTQ and AWQ behind one unified API (`GPTQConfig` / `AWQConfig`, same `load`/`quantize`/`save` calls), which is why `quantize_gptq_awq.py` barely branches on `--method`.

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-2-vertical-wins/08-quantization-ii-gptq-and-awq
pip install -r requirements.txt     # adds gptqmodel
python ingest.py                    # build corpus/ if you don't have it yet — today's calibration data
```

```bash
python quantize_gptq_awq.py --method gptq     # several minutes — go get coffee
python quantize_gptq_awq.py --method awq      # several minutes again
python benchmark_quantized.py --path qwen3-vl-8b-gptq-4bit
python benchmark_quantized.py --path qwen3-vl-8b-awq-4bit
```

## Troubleshooting

- **`gptqmodel` install/CUDA errors**: like `bitsandbytes`, it ships prebuilt kernels tied to specific CUDA versions — check `pip show gptqmodel` against your Studio's CUDA version if import fails.
- **Quantization errors on the vision tower**: `GPTQModel` documents quantizing "only text layers" for vision-language architectures (e.g. Qwen2-VL) as of its multi-modal support — if quantizing the full model errors out, check its docs for the current way to scope quantization to text-only modules for your `transformers`/`gptqmodel` version.
- **`corpus/chunks.json` not found**: run `python ingest.py` first — today's calibration data is Lecture 01's own manual corpus, not a downloaded dataset.
- **Quantization is much slower/faster than the lecture's ballpark**: calibration time scales with model size, calibration sample count (`N_CALIBRATION` in `quantize_gptq_awq.py`), and GPU — the lecture's timing is a ballpark, not a promise.
- Everything from Lectures 01–07's troubleshooting still applies if you also run those systems in this folder.

---

Previous: [`../07-quantization-i-number-formats/`](../07-quantization-i-number-formats/) · Next: [`../08b-build-the-eval-harness/`](../08b-build-the-eval-harness/)
