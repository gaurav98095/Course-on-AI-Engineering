---
name: blog-composer
description: Combine lesson blueprint, story, mental model, math, code, and visuals into a publish-ready Markdown article with a consistent lecture-style structure. Use when assembling the final AI engineering lesson draft.
---

# Blog Composer

## Purpose

Assemble the final lesson from modular stage outputs.

## Responsibilities

- Merge all stage artifacts.
- Preserve a consistent section order.
- Maintain heading hierarchy.
- Insert callouts, equations, tables, and exercises. No quiz or interview-question sections.
- Remove duplicated explanations.

## Inputs

- Lesson blueprint
- Story
- Mental model
- Math section
- Implementation section
- Visual brief

## Outputs

- Publish-ready Markdown article

## Rules

- Use one article template consistently.
- Keep the narrative flow smooth.
- Do not invent new lesson content unless needed to reconcile transitions.

## Examples

- Turn stage artifacts into a lecture-style article.
- Insert a mermaid figure directly after the section it supports.
- Keep the math and code explanations aligned.

## Failure Cases

- Duplicated explanations.
- Broken heading hierarchy.
- Missing transitions between sections.
- A final article that reads like concatenated notes.

## Checklist

- [ ] The article has a single coherent flow.
- [ ] Every required section is present.
- [ ] Headings are consistent.
- [ ] Callouts and details blocks are used well.
- [ ] The final Markdown is publish-ready.
