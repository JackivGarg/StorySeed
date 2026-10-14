# Standard for guided build notes

Apply the user's practical teaching contract in TEACHING_RULES.md to every new or revised phase.

## Write from the working system

Start with an immediate engineering goal and the concrete limitation of the previous step. Explain our approach before the code. Use actual StorySeed files and functions, then inspect a small example. Explain the decisions a student would have to make independently: dimensions, representation, label alignment, architecture, loss, device placement and update behavior.

The usual progression is goal/problem → approach → implementation → important code/data flow → observation → achievement/next step → debugging. Use natural headings and transitions instead of mechanically repeating eight headings.

## Code and evidence

- Link the real source and function. Excerpts must match the current code; say when a snippet is only the core excerpt.
- A standalone inspection example must include imports and all values it needs. Put the execution command nearby. For multi-line Python on this Windows setup, use a clearly described temporary local scratch file run from the code repository.
- A simplified numerical fixture must be labelled. It does not change the actual vocabulary or model.
- Trace the value and shape across important boundaries. Explain B (examples), T (positions), V (vocabulary choices), and any feature dimension at their first use. Do not give token IDs the meaning of learned features.
- Cite the report for recorded numbers. Label predicted or hand-calculated outcomes as expected, not as observed execution. Preserve existing reports when merely rewriting notes.
- Do not add code or experiments for decoration. Every inspection should diagnose a property or make an operation understandable.

## Engineering explanation

Explain what enters, what changes, what leaves and which module uses it next. Introduce the relevant theory while interpreting those operations. Define symbols and explain gradients/parameter updates where they occur. Describe a useful alternative briefly and justify the actual choice.

Debug through symptoms and localization: inspect representation and targets first, then score shapes, loss inputs, gradients, updates, evaluation and decoding. Demonstrate a concrete failing line or state and its correction when useful; do not only enumerate potential errors. State when a checkpoint supports inference but not exact training resume.

## Practical independence

A stage ends with a short "What I should now understand" list of things the student should be able to do. Companion practice is a small implementation/inspection session with a purpose, expectation, hints and worked repair or verification. Avoid routine exams, interview lists, MCQs, memorization and lengthy disconnected derivations.

Keep detail useful during coding. Several files may serve one subphase, but avoid repeated concept explanations. Revisit course concepts where they help this particular build. A runnable system and authored notes are not evidence of student mastery.

## Location and revision

Write lessons only in C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes/phaseXX. Use working absolute Markdown links to code/reports in the separate StorySeed folder, and relative links between notes. Update the external index and planning/LESSON_MAP.md. Preserve old notes in external history before content revisions, labelled with the code revision. The 2026-10-14 rewrite explicitly supersedes the copy-only relocation instruction; the originals remain archived. Do not recreate notes in the code repository.
