# AI Engineering Course Writing Instructions

Use this repository as a chapter-by-chapter writing framework for a publishable GitHub Pages course site. Each lecture page is the source material a YouTube video will be recorded from: it is shown on screen and read aloud as the lecture itself. Never add video links or references to a video inside the content.

The course: **advanced, hands-on AI engineering** — build a real multimodal RAG, deploy it, load-test it until it breaks, then spend the rest of the course making it 10× cheaper and scaling it to a lakh (100k+) concurrent users. Every lecture is a build log with numbers, the way a senior engineer ships at work.

Look and feel: clean, text-first pages with a narrow reading column, each lecture ending with a Next-lecture link. The design system (fonts, colors, section numbering, "Remember" chips) lives in `docs/_layouts/default.html`.

## Course arc (fixed — do not re-plan without being asked)

1. **Weeks 1–2 · Foundations — build it & measure it.** Multimodal RAG on stock Hugging Face code, deployed on **Lightning AI**, load-tested until it breaks; then the GPU and transformer fundamentals that explain every benchmark number (GPU architecture & HBM, roofline, prefill vs decode, KV cache math, TTFT/TPOT, profiling).
2. **Weeks 3–8 · Vertical wins — make it 10× faster.** Quantization (GPTQ/AWQ, FP8/INT8/INT4), FlashAttention, PagedAttention, GQA/MQA/MLA, RoPE/ALiBi/YaRN, continuous batching, vLLM & SGLang, TensorRT-LLM, speculative decoding, prefix & chunked prefill, Triton kernels, torch.compile & CUDA graphs — each win proven with a before/after benchmark and a quality eval. Still on Lightning AI.
3. **Weeks 9–12 · Modalities & scale — ship for 1L+ users.** Multimodal serving, embedding & reranker models, MoE, contrastive finetuning, Matryoshka embeddings, prefill/decode disaggregation, multi-LoRA; then the production move to **AWS**: Docker for GPUs, Kubernetes (EKS), autoscaling (HPA/KEDA), p95/p99 SLOs, $/Mtok and capacity math.

Platform arc: Lightning AI Studios for weeks 1–8 (fast iteration), AWS/EKS for the scale module — the same corporate progression a real team follows.

Course model registry (keep consistent across lectures): generation `Qwen/Qwen3-VL-8B-Instruct`; text embeddings `BAAI/bge-small-en-v1.5`; image–text embeddings `google/siglip2-base-patch16-224`; vector store FAISS. If a lecture swaps a model, it says so explicitly and explains why.

## Folder map

- `.claude/prompts/` — voice and teaching philosophy (read `professor_persona.md` first)
- `.claude/skills/` — modular writing stages (01–09)
- `.claude/workflows/` — deterministic pipelines
- `.claude/templates/` — chapter, math-page, and support templates
- `docs/` — the published site: `index.md`, `lectures/NN-slug.md`, `math/NN-slug.md`, `assets/images/`
- `code/module-N-<name>/NN-lecture-slug/` — one self-contained folder **per lecture**, nested under its module (data included); the lecture embeds excerpts and points at this exact folder. Copy-forward convention at both levels: a new lecture folder starts as a verbatim copy of the previous lecture's folder in the same module, and a new module's first lecture folder starts as a copy of the previous module's last lecture folder — each evolves from there. Any two folders can be diffed to see exactly what that lecture changed. See `code/README.md`.

## Writing a chapter

1. Read `.claude/prompts/professor_persona.md`.
2. Read the chapter brief or user request.
3. Run `.claude/workflows/workflow_chapter.md` — one stage at a time, using the skill files; do not write a monolithic answer.
4. Compose the final Markdown with `.claude/templates/chapter.md`; heavy derivations go to a separate `docs/math/NN-slug.md` page built from `.claude/templates/math_page.md`.
5. Publish with `.claude/workflows/workflow_publish.md` so the lecture gets its number, its row in `docs/index.md`, its previous/next navigation links, and its math page wired both ways.

## Rules

- One chapter per request. One primary story, one primary mental model, one clear math thread, one implementation thread.
- The Build section is the heart of every hands-on lecture: numbered steps, each with runnable code and "what you should see" output.
- **Say where every command runs.** The audience is general — they are following the site to build the project and must never guess the environment. Convention: 💻 laptop (browser only), ⚡ Lightning Studio terminal (the default for Modules 1–2; unlabeled blocks mean this), ☁️ AWS (Module 3, always labeled). Lecture 01's "where does each thing run?" table is the canonical form; every lecture that introduces a new environment repeats it.
- Ship the data. Every hands-on lecture's `code/<slug>/` folder must be self-contained from a fresh clone: input data (or a committed downloader with stable URLs), `requirements.txt`, a README with from-zero steps for a general audience, and expected outputs. Generated artifacts stay gitignored.
- Every benchmark number names its GPU, precision, and batch size. Numbers not actually measured are labelled ballpark ("on an L40S you should see roughly…"). Never invent a precise-looking number.
- Math delimiters: in the markdown SOURCE, always write inline math as `\\(...\\)` and display math as `\\[...\\]` — **double backslash**, not single. Never `$...$` or `$$...$$` (this site's prose quotes real dollar amounts like "~$1/hr", so a `$`-delimiter is unsafe; kramdown's own `$$` math-span recognition is also unreliable). And never a *single* `\(`/`\[` — kramdown treats parens/brackets as escapable punctuation and silently strips a lone backslash in front of them, which breaks MathJax with no visible error in the source. The double backslash is deliberate: kramdown's escape processor consumes `\\` and emits one literal `\`, leaving `\(`/`\[` intact in the rendered HTML, which MathJax then recognizes by default. Verify this rule wasn't violated by rendering a page and checking the browser, not just by reading the markdown source — the bug is invisible in source and only shows up after kramdown processes it.
- The same doubling rule applies *inside* any LaTeX environment that itself uses `\\` for a purpose — matrix/cases row separators (`\begin{pmatrix}`, `\begin{cases}`), forced line breaks. kramdown's escape processor consumes backslash pairs regardless of context, so a LaTeX row separator that needs MathJax to see `\\` (two backslashes) must be written as `\\\\` (four backslashes) in the markdown source — the ordinary "double it" rule only produces one surviving backslash, which breaks the row separator with no visible error. Prefer avoiding multi-row LaTeX environments when a simpler inline notation says the same thing; if you do use one, verify every row separator by rendering the page, the same way you'd verify any other delimiter. Also avoid non-base environments like `psmallmatrix` (a `mathtools` extension) — this site's MathJax config loads no extensions, so only base `amsmath`-style environments (`pmatrix`, `cases`, `matrix`, `align`) are guaranteed to render.
- **Underscores inside a math span can silently become markdown emphasis and vanish.** Kramdown parses the raw text of `\\(...\\)` / `\\[...\\]` for ordinary markdown syntax *before* MathJax ever sees it — it has no concept of "this is math, don't touch it." A subscript underscore that isn't flanked by word characters on both sides (e.g. `]_{F,q}`, `}_{\text{...}}`) reads to kramdown as a valid emphasis delimiter; when a math span (or even a whole paragraph) contains two or more such underscores, kramdown pairs them up, wraps the text between them in `<em>...</em>`, and the underscore characters themselves are consumed and gone from the output — MathJax then renders visibly broken LaTeX (stray HTML tags, missing subscripts). This shipped undetected in several already-published math pages until a live-site check caught it. Fix: escape every underscore inside a formula that has two or more of them as `\_` (single backslash — kramdown's normal escape rule for markdown special characters emits a bare literal `_` in the output, which is exactly what MathJax needs to see for a subscript; do **not** double it the way math delimiters are doubled, doubling here is the wrong rule for the wrong problem). A lone underscore in an otherwise underscore-free span is safe and doesn't need escaping, but when in doubt, escape it — it costs nothing and MathJax reads `\alpha\_n` and `\alpha_n` identically after kramdown processes the escape. Verify the same way as the other two delimiter bugs: render the page and check the browser (or fetch the live HTML and search for stray `<em>`/`<strong>` tags inside what should be a math span), never by re-reading the markdown source.
- Every hands-on lecture ends its build with a Measure It section — the benchmark table is the plot of the course; later lectures re-measure the same table after each optimization.
- Many small examples: at least three per lecture, built by `09_example_generator`, sharing one running example across math and code.
- Every paragraph must read aloud naturally — the page is a spoken script by a practitioner sharing knowledge, not a reference document.
- Presentation-first structure (the page is scrolled on screen while teaching):
  - Paragraphs are short beats — at most 3 sentences, often 1. A wall of text on screen kills a video.
  - Bold exactly the words a viewer should catch while the presenter talks; nothing else.
  - The single key claim of a section goes in a `>` blockquote (renders as a callout card).
  - Every concept section ends with a one-or-two-line takeaway paragraph tagged `{: .remember}` (renders as a "Remember" chip).
  - Big moments (a number, a flip, a punchline) get their own line.
- Use Mermaid and tables where they teach something.
- Include 2–4 real images per lecture (people, hardware, physical analogues from the story), sourced under free licenses, stored in `docs/assets/images/`, embedded with `<figure>` and an attribution caption. Never AI-generated images.
- Never leave a lecture unlinked: every published lecture points to the next one, the previous lecture points to it, and `docs/index.md` lists it. Math pages link back to their lecture and appear in the index's "Math deep dives" list.
- Use plain bold blockquotes, never GitHub `[!summary]`/`[!tip]` alerts — those render as literal text on GitHub Pages.

## Chapter writing contract

- Input: short chapter brief, desired depth, any prerequisites, and what to include (the creator's spoken description of the video).
- Output: publish-ready Markdown lecture in `docs/lectures/`, its math deep-dive page in `docs/math/` when the lecture has a math core, its runnable code in `code/<slug>/`, with all navigation wired.
- Do not invent a course plan — the arc above is fixed; `docs/index.md` is the source of truth for lecture order.
- Preserve consistency across chapters: notation, the course model registry, the running benchmark table, and the navigation contract.
- "Chapter", "lesson", "article", and "lecture" are synonyms; `workflow_chapter.md` and `templates/chapter.md` are the canonical files.
