---
name: example-generator
description: Generate a bank of small, concrete, mentally-simulable examples for a AI engineering lesson, from trivial to edge case. Use after the mental model is chosen so math, code, and visuals can reuse the same examples.
---

# Example Generator

## Purpose

Produce many tiny concrete examples so every abstract claim in the lesson lands on something the learner can compute in their head.

## Responsibilities

- Generate 3 to 7 examples per lesson, ordered from trivial to challenging.
- Use small numbers the learner can verify mentally (1 layer, 1 head, head dim 4, one token of cache).
- Reuse one running example across math, code, and visuals when possible.
- Include at least one counterexample or edge case that breaks naive intuition.
- Tag each example with where it belongs in the lesson (intuition, math, code, exercise).

## Inputs

- Lesson topic and blueprint
- Primary mental model
- Target audience level

## Outputs

- Example bank: numbered examples, each with setup, question, worked answer, and placement tag
- One designated running example
- One counterexample or edge case

## Procedure

1. Identify the 2 or 3 claims in the lesson that are hardest to believe abstractly.
2. For each claim, build the smallest concrete instance that demonstrates it.
3. Order examples so each one adds exactly one new complication.
4. Designate one example as the running example for math and code.
5. Add a counterexample that shows where the naive reading fails.
6. Verify every number by computing it, not by asserting it.

## Examples

- KV cache: 1 layer, 1 KV head, head dim 4, FP16 — count the 16 bytes per token by hand.
- Batching: two requests where one finishes at token 10 and one at token 500 — count the wasted slot-steps under static batching.
- Value iteration: a 3-cell gridworld small enough to run every sweep on paper.

## Failure Cases

- Examples too large to verify mentally.
- Every example at the same difficulty.
- Examples that use different notation than the math section.
- Wrong arithmetic in a worked answer.
- Examples that illustrate the analogy but not the actual mechanism.

## Checklist

- [ ] At least 3 examples, ordered by difficulty.
- [ ] Numbers are small and verified.
- [ ] One running example is designated.
- [ ] One counterexample or edge case is included.
- [ ] Each example has a placement tag.
- [ ] Notation matches the lesson's math section.
