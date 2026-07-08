# Workflow: Chapter Creation

Use this pipeline to produce one publish-ready AI engineering lecture. "Chapter", "lesson", "article", and "lecture" are synonyms in this course.

## Deterministic pipeline

1. `01_curriculum_architect` defines the chapter blueprint.
2. `02_story_builder` writes the opening story and motivation.
3. `03_mental_model_designer` defines one core intuition.
4. `09_example_generator` builds the example bank (running example, tiny examples, one edge case).
5. `04_math_teacher` develops the math section, reusing the running example.
   - When the lecture has a real mathematical core, `04_math_teacher` also produces a standalone deep-dive page from `templates/math_page.md` (full notation table, complete derivation, worked example, broken assumptions). The lecture keeps only the intuition, the final formula, one worked number, and a link card to the deep dive.
6. `05_code_teacher` builds the implementation section, reusing the running example.
7. `06_visual_designer` supplies diagrams, figure briefs, and 2–4 curated free-license photos saved to `docs/assets/images/`.
8. `07_blog_composer` assembles the final Markdown chapter, weaving the example bank through every section.
9. `08_editor` reviews the draft and returns fixes.
10. `workflows/workflow_publish.md` places the chapter on the site and wires the navigation links.

## Handoff contract

- Each stage receives only the upstream artifacts it needs.
- Each stage must produce a single artifact with a stable filename.
- Do not loop. If revisions are needed, update the relevant stage output once, then recompose.

## Artifact order

- `chapter_blueprint.md`
- `story.md`
- `mental_model.md`
- `examples_bank.md`
- `math.md`
- `math_page.md` (when the lecture has a math core)
- `implementation.md`
- `visuals.md`
- `chapter.md`
- `editor_review.md`

## Output bar

- The final chapter must be publish-ready Markdown that reads aloud as a lecture script.
- The final chapter must preserve one lesson, one story, one primary mental model, and one coherent flow.
- The final chapter must contain at least three concrete examples from the example bank.
- The final chapter must end with the navigation footer defined in `workflow_publish.md`.
