---
layout: default
title: "{{course_title}}"
---

# {{course_title}}

{{one_paragraph_course_pitch}}

## How to use this course

{{how_to_use — read each lecture in order, do the exercises, follow the Next link}}

## Lectures

<ol class="lectures">
  <li>
    <a href="lectures/NN-{{slug}}.html">   <!-- .html here: raw-HTML links are NOT rewritten by Jekyll -->
      <span class="num">NN</span>
      <span>
        <span class="t">{{lecture_title}}</span>
        <span class="d" style="display:block">{{one_line_hook}}</span>
      </span>
      <span class="go">→</span>
    </a>
  </li>
</ol>

## What you will be able to do

- {{outcome_1}}
- {{outcome_2}}
- {{outcome_3}}

---

Start here: [Lecture 01 — {{lecture_1_title}} →](lectures/01-{{slug_1}}.md)
