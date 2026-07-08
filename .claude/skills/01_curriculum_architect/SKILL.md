---
name: curriculum-architect
description: Plan AI engineering lesson sequences, prerequisites, dependency graphs, learning objectives, lecture duration, and difficulty. Use when designing or restructuring a course, module, syllabus, or lesson blueprint.
---

# Curriculum Architect

## Purpose

Design the teaching plan before any drafting begins.

## Responsibilities

- Define lesson scope.
- Identify prerequisites.
- Write learning objectives.
- Build the lesson dependency graph.
- Estimate lecture duration.
- Assign difficulty.

## Inputs

- Course theme
- Audience profile
- Existing outline, if any
- Desired lesson depth

## Outputs

- Structured lesson blueprint
- Dependency graph
- Prerequisite list
- Duration estimate
- Difficulty rating

## Procedure

1. Determine the smallest coherent lesson unit.
2. List only the prerequisites the learner truly needs.
3. Write objectives as observable outcomes.
4. Order the concepts by dependency, not by author preference.
5. Estimate time from explanation load, not page count.
6. Mark the difficulty honestly.

## Examples

- Plan a first lesson on Markov Decision Processes.
- Split quantization into two lectures when the derivation is too dense.
- Add a prerequisite on probability when advantage estimation depends on expectation notation.

## Failure Cases

- Circular dependencies.
- Hidden prerequisites.
- Overstuffed lessons.
- Vague objectives like "understand AI engineering."
- Duration estimates that ignore derivation or coding time.

## Checklist

- [ ] Lesson has one clear scope.
- [ ] Prerequisites are explicit.
- [ ] Objectives are measurable.
- [ ] Dependencies are acyclic.
- [ ] Duration is plausible.
- [ ] Difficulty is labeled.
