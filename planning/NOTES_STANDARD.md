# Notes delivery standard

Our notes are the student's main learning resource. Write detailed, connected chapters alongside the implementation. Do not substitute brief definitions, a checklist, external links or unexplained code for a lesson.

For each substantive chapter:

1. Identify phase/subphase, previous working state and learning question. Link actual prior chapters and relevant code.
2. Explain the concrete limitation that motivates the next idea and why it belongs at this point.
3. Revise relevant prior concepts with an example. Explain similarities and differences; do not dismiss them as already known.
4. Develop intuition and a hand-worked numerical example for major numerical ideas. Define all symbols, axes and assumptions.
5. Derive the relevant formula and interpret each step. Trace input/intermediate/output shapes. Distinguish IDs, vectors, logits, probabilities, labels, weights, loss and gradients.
6. Walk through important code in execution order, relating expressions to the calculation. Explain unfamiliar Python and library calls.
7. Explain training versus inference, learned/fixed values and gradient/update behavior when relevant.
8. Give the actual run command, relevant setup and observed outputs; interpret evidence without inventing measurements.
9. Discuss common mistakes, their symptoms and useful alternative choices. Use bounded experiments when they answer a specific question.
10. Provide a few guided exercises, hints and worked solutions, separated so the student can try first.
11. Explain the remaining limitation and next dependency. Record understanding as pending until the student's own evidence arrives.
12. Include relevant primary references and vocabulary. Optional sources support the chapter rather than replace it.

Use multiple files for different learning questions: a concept chapter, worked examples, a code walkthrough and practice/solutions when useful. There is no compulsory number of files or word limit. Length serves understanding; repeated filler does not.

A setup lesson needs a clear runnable environment explanation, not an invented mathematical derivation. Attention needs worked matrix calculations and shape tracing. The same principle adapts depth to each concept.

Canonical chapters belong in C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes/phaseXX. The user requested a separate Desktop folder and exact copies of the existing files. Keep the external reading index ordered and link companion files; map lessons to their code increments in planning/LESSON_MAP.md. Do not place teaching chapters in this code repository. Fix future code/lesson discrepancies truthfully. Do not mark templates or planned chapter titles as delivered lessons.

Older project notes remain reference material. New chapters will be written in StorySeed-Study-Notes against the actual implementation in StorySeed. Preserve previous lesson versions in the external folder before revising their content, labelled with the corresponding code commit, so historical study remains possible.
