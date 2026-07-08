---
layout: default
title: "Lecture {{lecture_number}} — {{chapter_title}}"
---

# Lecture {{lecture_number}} — {{chapter_title}}

> **In one sentence:** {{one_sentence_why_it_matters}}

<!-- Use plain bold blockquotes, never GitHub [!summary]/[!tip] alerts — those render as literal text on GitHub Pages.
     MATH DELIMITERS: always write inline math as \(...\) and display math as \[...\] — never $...$ or $$...$$.
     Kramdown's $-based math parsing is unreliable (it inconsistently converts, or silently leaves LaTeX as literal
     text), and this site's prose also quotes real dollar amounts ("~$1/hr"), so a $-based delimiter is unsafe.
     \(\)/\[\] are passed through untouched by kramdown and are MathJax's default delimiters — no config needed. -->

## Learning Objectives

- {{objective_1}}
- {{objective_2}}
- {{objective_3}}

## Prerequisites

| Concept | Needed? | Notes |
| --- | --- | --- |
| Python | Yes | {{python_note}} |
| {{prereq_2}} | {{prereq_2_needed}} | {{prereq_2_note}} |
| {{prereq_3}} | {{prereq_3_needed}} | {{prereq_3_note}} |

## Story

{{story_intro}}

<figure>
  <img src="../assets/images/{{story_photo}}" alt="{{story_photo_alt}}">
  <figcaption>{{story_photo_teaching_sentence}} <em>Photo: {{story_photo_attribution}}</em></figcaption>
</figure>

## Mental Model

{{primary_analogy}}

{{one_line_takeaway}}
{: .remember}

<!-- End every concept section with a {: .remember} chip like the one above.
     Keep paragraphs to at most 3 sentences — the page is presented on screen. -->

## The System

{{architecture_walkthrough}}

```mermaid
{{architecture_diagram}}
```

## The Build

<!-- The heart of the lecture: numbered steps. Each step = goal sentence, code block,
     then "What you should see" with real or clearly-ballpark output. -->

### Step 1 — {{step_1_title}}

{{step_1_goal}}

```python
{{step_1_code}}
```

{{step_1_what_you_should_see}}

### Step 2 — {{step_2_title}}

{{...}}

## Measure It

<!-- Every lecture ends its build with numbers. Name the hardware. -->

{{measurement_walkthrough}}

| Metric | Value | Conditions |
| --- | --- | --- |
| {{metric_1}} | {{value_1}} | {{conditions_1}} |

## The Math, One Level Deeper

{{math_intuition_and_final_formula_only}}

\[
{{key_equation}}
\]

{{worked_running_example}}

> **Want the full derivation?** Every symbol, every step, and where the assumptions break:
> [Math Deep Dive — {{math_page_title}} →](../math/{{math_page_file}})

## Where It Breaks

{{edge_case_or_failure_mode_from_example_bank}}

## Exercises

1. {{exercise_1}}
2. {{exercise_2}}
3. {{exercise_3}}

## Summary

{{summary}}

> **What should you remember?**
> - {{takeaway_1}}
> - {{takeaway_2}}

## Resources

- {{reading_1}}
- {{reading_2}}

---

[← Previous: Lecture {{prev_number}} — {{prev_title}}]({{prev_file}}) · [Course Home](../index.md) · [Next: Lecture {{next_number}} — {{next_title}} →]({{next_file}})
