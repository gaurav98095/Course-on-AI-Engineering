# Workflow: Publish a Lecture to the Site

Use this workflow every time a finished lecture Markdown file must join the GitHub Pages site. Run it after `workflow_chapter.md` produces the final draft, or standalone when re-linking existing lectures.

## Site layout

```
docs/
  index.md                  <- course landing page (lecture list, in order)
  lectures/
    01-build-a-multimodal-rag.md
    02-deploy-it-on-a-gpu.md
    ...
  math/
    01-geometry-of-retrieval.md   <- deep-dive pages, numbered after their lecture
    ...
  assets/                   <- figures, css if any
```

- GitHub Pages serves from `docs/` on the main branch.
- Filenames follow `NN-kebab-slug.md`. `NN` is the zero-padded lecture number and never changes once published.
- Relative `.md` links are used everywhere; GitHub Pages rewrites them to `.html` automatically.

## Navigation contract

Every lecture file must end with this footer, separated by a horizontal rule:

```markdown
---

[← Previous: Lecture NN — Title](NN-slug.md) · [Course Home](../index.md) · [Next: Lecture NN — Title →](NN-slug.md)
```

- The first lecture omits the Previous link. The latest lecture omits the Next link.
- The lecture titles in the footer must match the target files' H1 titles exactly.

## Math deep-dive pages

- A math page lives at `docs/math/NN-slug.md`, where `NN` matches its lecture's number.
- The lecture links to it with the "Want the full derivation?" blockquote card from `templates/chapter.md`.
- The math page footer links back to its lecture and to the course home (no prev/next chain).
- Every math page is listed in the "Math deep dives" section of `docs/index.md`.

## Pipeline

1. Read `docs/index.md` to learn the current lecture order.
2. Assign the new lecture its number and slug; save it to `docs/lectures/NN-slug.md`.
3. Ensure the file has the frontmatter and section structure required by `templates/chapter.md`.
4. Append the navigation footer to the new lecture (no Next link yet).
5. Edit the previous lecture's footer: add the Next link pointing at the new file.
6. Add the new lecture to the `<ol class="lectures">` list in `docs/index.md`: number, title, a one-line hook, and an `.html` href (raw-HTML links are not rewritten by Jekyll, so use `.html` there — unlike Markdown links, which use `.md`).
7. If the lecture ships a math deep-dive page, save it to `docs/math/NN-slug.md`, add its back-link footer, insert the link card into the lecture, and add it to the "Math deep dives" list in `docs/index.md`.
8. If the lecture ships runnable code, place it in `code/<lecture-slug>/` with a `requirements.txt`; the lecture embeds excerpts but the folder must run standalone.
9. Verify every link touched above points at a file that exists.

## Rules

- Never renumber published lectures. Insertions get a suffix (e.g. `03b-`) or go at the end.
- Never leave a dangling Next link: the previous lecture is updated in the same run as the new one.
- `docs/index.md` is the single source of truth for lecture order.

## Output

- New lecture file in `docs/lectures/`
- Updated previous lecture (footer only)
- Updated `docs/index.md`
- A short report listing the three files touched and the links verified
