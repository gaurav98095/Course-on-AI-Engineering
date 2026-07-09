# Lecture 08b — Build the Eval Harness

Self-contained: this folder is a **copy-forward** of `08-quantization-ii-gptq-and-awq/` plus three new files. Clone the repo, `cd` here, and everything runs.

Full walkthrough: [Lecture 08b on the course site](https://gaurav98095.github.io/Course-on-AI-Engineering/lectures/08b-build-the-eval-harness.html)

## Where does everything run?

| Environment | Role in this lecture |
| --- | --- |
| ⚡ Lightning AI Studio | Everything — `build_eval_set.py` needs the GPU-backed retriever, `eval.py --generate` needs the GPU-backed generator |
| 💻 Your laptop | Browser only, reading the lecture |
| ☁️ AWS | Nothing yet — Module 3 |

## What's new versus Lecture 08

| File | What it is |
| --- | --- |
| `eval_set.json` | 10 question → answer-key pairs, reusing `load_test.py`'s question list. `expected_page` ships as `null` — a real answer key has to be verified, not guessed, and this repo isn't going to fabricate page numbers it never checked |
| `build_eval_set.py` | Runs retrieval for every unconfirmed question, shows you the real hits, and lets you confirm which page is actually correct — that's what turns the template into a real eval set |
| `eval.py` | Runs the confirmed eval set against any checkpoint: retrieval recall@k always, and (with `--generate`) full RAG answer-quality — required-term coverage and citation accuracy |

`rag.py`'s `Generator` gained one optional argument, `quantized_path`, so `eval.py` can point it at a GPTQ/AWQ checkpoint from Lecture 08 instead of stock bf16.

(Everything from Lectures 01–08 is copied forward unchanged, so this folder still runs the full system, and both quantization scripts, on its own.)

## Step by step from zero

```bash
git clone https://github.com/gaurav98095/Course-on-AI-Engineering.git
cd Course-on-AI-Engineering/code/module-2-vertical-wins/08b-build-the-eval-harness
pip install -r requirements.txt
python ingest.py                    # build corpus/ if you don't have it yet
```

```bash
python build_eval_set.py                                  # confirm the answer key once, interactively
python eval.py                                             # retrieval recall@k, bf16, fast
python eval.py --generate                                  # + answer quality, bf16
python eval.py --generate --checkpoint qwen3-vl-8b-gptq-4bit   # same eval, quantized generator (needs Lecture 08's quantize_gptq_awq.py run first)
```

## Troubleshooting

- **`eval.py` exits immediately with "N question(s) still have no confirmed answer"**: run `build_eval_set.py` first — `eval.py` refuses to score against an unverified answer key rather than silently produce a meaningless number.
- **`build_eval_set.py`'s retrieved hits all look wrong for a question**: that's real signal, not a bug — either `ingest.py` didn't index the right chapter, or the question doesn't have a good answer in this corpus. Type `s` to skip it and note it as a corpus gap.
- **`--checkpoint` path fails to load**: it needs a folder saved by Lecture 08's `quantize_gptq_awq.py` (e.g. `qwen3-vl-8b-gptq-4bit/`) — run that first if you haven't.
- **Term-coverage and citation-accuracy numbers look harsh**: both checks are intentionally literal (exact substring match, exact `[doc p.N]` citation match) — a correct answer phrased differently, or citing an equally-valid alternate page, will register as a miss. That's a real limitation of this harness, not a bug; see Where It Breaks in the lecture.
- Everything from Lectures 01–08's troubleshooting still applies if you also run those systems in this folder.

---

Previous: [`../08-quantization-ii-gptq-and-awq/`](../08-quantization-ii-gptq-and-awq/) · Next: [`../09-flashattention/`](../09-flashattention/)
