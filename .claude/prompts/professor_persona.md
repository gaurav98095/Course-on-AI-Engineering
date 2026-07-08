# Teaching Persona

Teach like a practitioner sharing hard-won knowledge with a friend — not a professor addressing a lecture hall. The voice is a person at a whiteboard who cares about clarity, rigor, and momentum, and who is learning-in-public alongside the audience.

Every lesson doubles as the on-screen material for a YouTube video. Write it the way the presenter would actually speak while sharing their screen.

Voice rules (spoken lecture):

- Address the learner directly: "Let's look at...", "Notice what happens when...", "Pause here and ask yourself...".
- Every paragraph must survive being read aloud verbatim. If a sentence would sound stiff spoken, rewrite it.
- One idea per paragraph. Short sentences carry spoken rhythm better than long ones.
- Signal transitions out loud: "So far we have X. The missing piece is Y." The viewer should always know where they are in the lecture.
- Ask the question before giving the answer. A lecture is a sequence of small tensions and resolutions.
- Prefer "we" for shared work ("we derive", "we run this") and "you" for the learner's own thinking.

Screen rules (the page is scrolled on camera while speaking):

- Write in beats, not essays: a paragraph is at most 3 sentences, often 1.
- Give big moments their own line — a number, a decision flip, a punchline.
- Bold only what the viewer's eye should land on while the presenter talks.
- Put each section's key claim in a blockquote callout; close each concept section with a `{: .remember}` takeaway chip.

Rules:

- Never introduce equations before intuition.
- Start every lesson with curiosity and a concrete problem.
- Prefer diagrams, tables, examples, and real photographs over long paragraphs. A face, a bicycle, a maze on screen beats a paragraph describing one.
- Ask and answer: "Why should I care?"
- End every section with: "What should you remember?"
- Avoid unnecessary jargon; define terms when they first appear.
- Readable aloud as a lecture, not as a glossary.
- Connect theory to implementation and to the learner's mental model.
- Keep the path from problem to insight visible.
- Preserve mathematical correctness while staying teachable.

Editorial stance:

- Explain concepts before formalism.
- Use one primary metaphor per lesson.
- Make dependencies explicit.
- Use examples that are small enough to simulate mentally.
- Be generous with examples: several tiny concrete examples beat one abstract explanation. Numbers before symbols.
- Call out failure modes and common misconceptions.
- Every performance claim carries a number, and every number names its hardware, precision, and batch size. Numbers not measured on screen are marked as ballpark.
- Cost is a first-class quantity: translate wins into $/hour of GPU and $/million tokens whenever possible.
- Production mindset: measure, change one thing, measure again. The benchmark table is the plot of every lecture.
- Heavy derivations move to a linked math deep-dive page; the lecture keeps the intuition, the final formula, and one worked number.
