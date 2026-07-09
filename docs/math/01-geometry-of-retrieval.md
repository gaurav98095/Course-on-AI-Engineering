---
layout: default
title: "Math Deep Dive — The Geometry of Retrieval"
---

# Math Deep Dive — The Geometry of Retrieval

> **Supports:** [Lecture 01 — Build a Multimodal RAG](../lectures/01-build-a-multimodal-rag.md). Read the lecture first; this page assumes its story and mental model.

## The Idea in One Picture

An embedding model maps meaning to **direction**. Every chunk, every figure, every question becomes an arrow from the origin in a \\(d\\)-dimensional space, and the model is trained so that *things that mean the same point the same way*.

Once you accept that, retrieval stops being mysterious. "Find relevant chunks" becomes "find arrows with a small angle to the question's arrow" — and everything on this page is just making *small angle* precise, fast, and trainable.

## Notation

| Symbol | Meaning | Typical value |
| --- | --- | --- |
| \\(\mathbf{a}, \mathbf{b}\\) | embedding vectors | — |
| \\(d\\) | embedding dimension | 384 (BGE-small), 768 (SigLIP 2 base) |
| \\(\lVert\mathbf{a}\rVert\\) | Euclidean length, \\(\sqrt{\sum_i a_i^2}\\) | 1 after normalization |
| \\(\theta\\) | angle between two vectors | — |
| \\(\mathbf{a}\cdot\mathbf{b}\\) | dot product, \\(\sum_i a_i b_i\\) | in \\([-1, 1]\\) for unit vectors |
| \\(\tau\\) | temperature in the contrastive loss | ~0.01–0.1 (learned, in CLIP) |
| \\(N\\) | pairs per training batch | 32k in CLIP |
| \\(k\\) | results returned per query | 4 text + 2 images in our build |

## Derivation

### Part 1 — Why normalize? Dot, cosine, and Euclidean all agree

The dot product has a geometric identity — this is the definition of the angle between vectors:

\\[
\mathbf{a}\cdot\mathbf{b} = \lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert\cos\theta
\\]

Divide both sides by the lengths, and cosine similarity is just "the dot product with the lengths cancelled":

\\[
\cos\theta = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}
\\]

Now normalize every vector to unit length: \\(\lVert\mathbf{a}\rVert = \lVert\mathbf{b}\rVert = 1\\). The denominator becomes 1, and:

\\[
\cos\theta = \mathbf{a}\cdot\mathbf{b}
\\]

The *cheapest* operation (\\(d\\) multiplies and adds) and the *right* operation (angle) are now the same number. That is what `normalize_embeddings=True` and `IndexFlatIP` conspire to exploit.

Euclidean distance joins the party too. Expand the squared distance between unit vectors:

\\[
\lVert\mathbf{a}-\mathbf{b}\rVert^2
= \lVert\mathbf{a}\rVert^2 - 2\,\mathbf{a}\cdot\mathbf{b} + \lVert\mathbf{b}\rVert^2
= 2 - 2\,\mathbf{a}\cdot\mathbf{b}
\\]

Distance is a decreasing function of the dot product — nothing else. So on the unit sphere, **ranking by dot product, by cosine, and by Euclidean distance gives the identical order**. There is no "which metric should I use?" decision left to make.

### Part 2 — Where do these spaces come from? Contrastive training

Nothing so far explains *why* a question about stalls points the same way as a paragraph about stalls. That is manufactured by training, and the recipe is contrastive.

Take a batch of \\(N\\) matched pairs — for CLIP, images with their captions. Encode images with one tower, texts with another, normalize, and compute all \\(N\times N\\) similarities \\(s_{ij} = \mathbf{u}_i \cdot \mathbf{v}_j\\). Pair \\((i,i)\\) is the match; every \\((i,j),\, j\neq i\\) is a mismatch.

CLIP treats each row as an \\(N\\)-way classification problem — "which of these \\(N\\) captions is mine?" — with softmax cross-entropy:

\\[
\mathcal{L}\_i = -\log \frac{e^{\,s\_{ii}/\tau}}{\sum\_{j=1}^{N} e^{\,s\_{ij}/\tau}}
\\]

Read the fraction. To make the loss small, the numerator (\\(s_{ii}\\), the matched similarity) must rise, and the denominator (all the mismatched similarities) must fall. Gradient descent therefore **pulls matched image–caption arrows together and pushes everything else apart** — millions of batches later, "meaning = direction" is simply true, across modalities, because the loss only ever spoke about cross-modal dot products.

The temperature \\(\tau\\) divides every similarity before the softmax, so small \\(\tau\\) magnifies differences. Concretely, with our lecture scores \\(0.96\\) vs \\(0.28\\):

- \\(\tau = 1\\): softmax of \\((0.96, 0.28)\\) → probabilities \\((0.66,\ 0.34)\\). Mild preference.
- \\(\tau = 0.07\\) (CLIP-like): softmax of \\((13.7,\ 4.0)\\) → \\((0.99994,\ 0.00006)\\). Near-certainty.

Same geometry, very different pressure on the gradients — \\(\tau\\) controls how hard the model is punished for near-misses.

**SigLIP's twist** — and why we used it — is replacing the batch-wide softmax with an independent sigmoid per pair:

\\[
\mathcal{L}\_{ij} = \log\!\left(1 + e^{\,z\_{ij}\,(-t\, s\_{ij} + b)}\right),
\qquad z\_{ij} = \begin{cases} +1 & i = j\\\\ -1 & i \neq j\end{cases}
\\]

Each pair is now a standalone yes/no question — "do these two belong together?" — so the loss no longer needs every other example in the batch to normalize against. That makes training scale gracefully, and it is why the SigLIP family gives such strong open checkpoints for us to build on.

## Worked Example

The lecture's three unit vectors, taken all the way through every formula above. Question \\(q = (0.6, 0.8)\\); chunk \\(c_1 = (0.8, 0.6)\\); chunk \\(c_2 = (-0.6, 0.8)\\). (Check: \\(0.6^2 + 0.8^2 = 1\\) — all three are unit length.)

**Dot products:**

\\[
q\cdot c_1 = 0.48 + 0.48 = 0.96 \qquad q\cdot c_2 = -0.36 + 0.64 = 0.28
\\]

**Euclidean check** via \\(\lVert q - c\rVert^2 = 2 - 2\,q\cdot c\\):

\\[
\lVert q - c_1\rVert^2 = 2 - 1.92 = 0.08 \qquad \lVert q - c_2\rVert^2 = 2 - 0.56 = 1.44
\\]

Same winner, same order — exactly as Part 1 promised. FAISS with `IndexFlatIP` computes the two dot products (in general, \\(N \times d\\) multiply-adds for \\(N\\) cards) and returns the top-\\(k\\) indices. At our corpus size that is ~500 × 384 ≈ 200k multiply-adds per query — effectively free next to a single token of an 8B-parameter model.

**And the metric that judges it all** — recall@k, from Exercise 4 of the lecture:

\\[
\text{recall@}k = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}\!\left[\text{a correct document ranks in } q\text{'s top-}k\right]
\\]

It deliberately ignores the generator: retrieval gets measured alone, exactly the way we built it alone in Step 4.

## Where the Assumptions Break

**The modality gap.** In real CLIP/SigLIP models, image embeddings and text embeddings each occupy their own *cone* of the sphere — matched pairs are close *relative to mismatches*, but images stay measurably offset from texts as a population. Cross-modal ranking still works (the offset is roughly shared), but absolute image–text scores run lower than text–text scores. One more reason the lecture keeps separate top-k quotas per index.

**Scores don't transfer between models.** Each model's training temperature and data shape its score distribution. BGE's "0.71" and SigLIP's "0.31" are answers to different exam papers — comparable within an index, meaningless across.

**The caption-length ceiling.** SigLIP's text tower saw ≤64-token captions in training; it doesn't just truncate longer inputs, it was never taught what long text *means*. Long-document embedding needs models trained for it — that is BGE's job today, and our own finetuned model's job in Module 3.

**High dimensions are weird.** In 384-d, random unit vectors are almost always nearly orthogonal, and some points become *hubs* that show up in everyone's top-k. Empirically embeddings still work — but keep healthy suspicion of raw similarity values; trust rankings and measured recall instead.

## Common Mistakes

- **Skipping normalization with `IndexFlatIP`.** Unnormalized inner product rewards long vectors, so you rank partly by vector length — a training artifact — not by angle. Symptom: the same few documents dominate every query.
- **Comparing scores across two indexes** (or merging their rankings by score). Different models, different rulers. Use per-index quotas, or a reranker.
- **Reading cosine 0.9 as "90% similar".** Cosine is not a probability or a percentage; its useful scale is relative, per model. 0.9 might be a mediocre match for one model and outstanding for another.
- **Embedding long text with a caption model** (or vice versa). Match the encoder to the distribution it was trained on — the tokenizer will not warn you.

---

[← Back to Lecture 01 — Build a Multimodal RAG](../lectures/01-build-a-multimodal-rag.md) · [Course Home](../index.md)
