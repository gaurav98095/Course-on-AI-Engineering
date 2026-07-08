# Lecture 01 — Build a Multimodal RAG

Self-contained: clone the repo, `cd` into this folder, and everything runs — **including the data**. It answers questions about the FAA *Pilot's Handbook* (a real illustrated manual, public domain) by retrieving text **and figures**, then reading both with a vision-language model.

Full walkthrough: [Lecture 01 on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/01-build-a-multimodal-rag.html)

## Where does everything run?

| Environment | What happens there | In this lecture |
| --- | --- | --- |
| 💻 **Your laptop** | Browser only: reading the course, clicking around Lightning AI | No code runs here |
| ⚡ **Lightning AI Studio** | A cloud machine with a GPU that you control through a browser. **Every command below runs in its terminal.** | Everything |
| ☁️ **AWS EC2 / EKS** | Where the system goes in Module 3 | Nothing yet |

## What's in this folder

| File | What it is |
| --- | --- |
| `data/phak-ch4-aerodynamics.pdf` | Corpus part 1: Aerodynamics chapter (FAA, public domain) |
| `data/phak-ch7-instruments.pdf` | Corpus part 2: Flight Instruments chapter (FAA, public domain) |
| `data/sample-query-instrument.jpg` | A cockpit instrument photo (public domain) for image-as-query |
| `requirements.txt` | Python dependencies for this lecture |
| `ingest.py` | PDFs → text chunks + figures → two FAISS indexes |
| `rag.py` | Retrieve + generate: ask a question, get a cited answer |
| `measure.py` | The baseline benchmark: TTFT, TPOT, tok/s, VRAM |

Generated output (`corpus/`) is gitignored — rebuild it in ~2 minutes with `ingest.py`.

## Step by step from zero

### 1 · 💻 On your laptop — get a GPU machine

1. Sign up at [lightning.ai](https://lightning.ai) (free tier includes GPU credits).
2. Create a **Studio** and switch its hardware to a GPU: **L40S (48 GB)** recommended, an **L4 (24 GB)** works but is tight.
3. Open the Studio — you get a browser terminal. Everything from here happens there.

### 2 · ⚡ In the Studio terminal — get the code

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-1-foundations/01-build-a-multimodal-rag
pip install -r requirements.txt
nvidia-smi        # confirm the GPU is really there
```

### 3 · ⚡ Build the indexes

```bash
python ingest.py
```

```text
phak-ch4-aerodynamics: ~230 text chunks, ~60 figures
phak-ch7-instruments:  ~260 text chunks, ~90 figures
indexed ~490 chunks + ~150 figures -> corpus/
```

First run also downloads the two embedding models (~1 GB) from Hugging Face.

### 4 · ⚡ Ask questions

```bash
python rag.py "Why does an aircraft stall at the critical angle of attack?"
```

First run downloads Qwen3-VL-8B (~16 GB) — one-time, a few minutes.

Ask **with a picture** (sample ships in `data/`):

```bash
python rag.py "What does this instrument do and what errors does it have?" \
  --image data/sample-query-instrument.jpg
```

### 5 · ⚡ Record your baseline

```bash
python measure.py
```

Save the table. Ballpark on one L40S, bf16, batch 1: TTFT ~1 s, ~30 tok/s, ~22 GiB peak VRAM, one user at a time.

### 6 · 💻 When you're done

Stop the Studio from the Lightning dashboard — a stopped Studio doesn't burn GPU credits.

## Use your own documents

Drop any illustrated PDF into `data/` and rerun `python ingest.py` — every PDF there gets indexed.

## Troubleshooting

- **CUDA out of memory** on a 24 GB card: in `rag.py`, use `k_img=1` and `max_new_tokens=200`.
- **Slow model download**: `pip install hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1`, then retry.
- **`faiss` import error**: make sure it's `faiss-cpu` — the index is tiny, GPU FAISS is unnecessary here.

## Licenses

FAA handbook chapters: US Government work, public domain. Sample photo: public domain (Wikimedia Commons). Code: same license as the repository.

---

Next: [`../02-deploy-it-on-a-gpu/`](../02-deploy-it-on-a-gpu/) — this same system, wrapped in an HTTP endpoint.
