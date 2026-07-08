# Examples

## Example 1

Topic: Prefill vs decode

Output:

- Mermaid diagram: prompt tokens flowing in parallel into prefill, then a one-token-at-a-time decode loop feeding back
- Caption explaining why one phase is a burst and the other a drip

## Example 2

Topic: Roofline model

Output:

- SVG description: bandwidth slope and compute ceiling meeting at the ridge point, with decode plotted left and prefill right
- Animation suggestion: a workload dot sliding right as batch size grows

## Example 3

Topic: Serving architecture

Output:

- Mermaid diagram of client → load balancer → engine (scheduler, KV cache pool) → GPU
- Caption tracing one request through the queue

## Example 4

Topic: Lecture 01 story beats (multimodal RAG, the librarian, the GPU)

Output:

- Photo: library card catalog drawers — Wikimedia Commons, free license → `docs/assets/images/card-catalog.jpg`, placed inside the opening story
- Photo: NVIDIA data-center GPU board — Wikimedia Commons, free license → `gpu-board.jpg`, placed beside the first deployment section
- Diagram: end-to-end RAG pipeline SVG → `rag-pipeline.svg`, placed in the architecture section and referenced again by the code walkthrough
