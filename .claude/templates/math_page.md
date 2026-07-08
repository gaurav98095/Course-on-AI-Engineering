---
layout: default
title: "Math Deep Dive — {{math_title}}"
---

# Math Deep Dive — {{math_title}}

> **Supports:** [Lecture {{lecture_number}} — {{lecture_title}}](../lectures/{{lecture_file}}). Read the lecture first; this page assumes its story and mental model.

<!-- MATH DELIMITERS: in the markdown SOURCE, write inline math as \\(...\\) and display math as \\[...\\] —
     DOUBLE backslash, not single, and never $...$/$$...$$. See CLAUDE.md and templates/chapter.md for why:
     kramdown silently strips a lone backslash before parens/brackets, breaking MathJax with no visible error
     in the source. Verify by rendering, not by re-reading the markdown. -->

## The Idea in One Picture

{{intuition_recap_short}}

## Notation

| Symbol | Meaning | Typical value |
| --- | --- | --- |
| {{symbol_1}} | {{meaning_1}} | {{typical_1}} |

## Derivation

<!-- Step by step. Never skip algebra. Every step gets one sentence of justification. -->

{{derivation}}

## Worked Example

<!-- Reuse the lecture's running example with real numbers. -->

{{worked_example}}

## Where the Assumptions Break

{{assumption_failures}}

## Common Mistakes

- {{mistake_1}}
- {{mistake_2}}

---

[← Back to Lecture {{lecture_number}} — {{lecture_title}}](../lectures/{{lecture_file}}) · [Course Home](../index.md)
