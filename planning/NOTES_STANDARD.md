# Notes standard — concept and reasoning first, connected to the build

Follow TEACHING_RULES.md, updated for the user's 2026-10-15 feedback.

## A chapter starts from an unresolved project need

Explain the immediate goal within the larger SLM journey. Show why the current system cannot yet accomplish it. Consider a plausible approach, then identify the concept or prerequisite needed to assess or implement that approach.

Follow the conceptual dependency: project need → reasoning/choice → prerequisite learning → explicit return to the project → justified implementation → observation/limitation → next dependency. The source-file order is not the teaching syllabus.

## Teach the mechanism before a code tour

Develop the concept with intuition, concrete examples and appropriate mathematics. Define symbols and tensor axes when needed. State what is adjustable, fixed, observed or computed. Explain why each transformation is necessary and how it supports the next component.

A small snippet may reveal a concept. Do not begin by importing a project function, running it, and explaining its lines as the main lesson. The student should be able to follow the reasoning without opening the finished code first.

Do not make every paragraph a definition or a rigid checklist. Give prerequisites space and then reconnect them: “we paused this component to understand X; now X lets us decide/do Y.”

## Scope and depth

Assume basic Python and limited practical deep learning fluency, despite prior lecture exposure. Skip redundant elementary token/Python explanations. Teach unfamiliar PyTorch, differentiation, representation and architecture ideas carefully where they belong. There is no fixed maximum chapter length; multiple files are welcome when connected and useful.

Preserve a continuing concrete prediction example where possible. Separate conceptual illustrations from actual model behavior. Explain alternatives briefly and why our project chooses one.

## Code and observations

Reference real files/functions after their purpose is understood. Complete runnable illustrations need imports and all required values; excerpts must be identified. Explain important operations semantically, not just syntactically.

Use actual report links for observed values. Hand calculations and predicted results are labelled as expectations. Do not run or invent a new training job to support an editorial rewrite. Do not change functioning model source for presentation.

Trace data values/shapes as part of explaining the mechanism, not as a replacement for the reasoning. Distinguish IDs, parameters, features, scores, probabilities, loss, gradients and updates when they occur.

## Understanding and debugging

Show how conceptual understanding locates a failure: wrong examples, incompatible representation, inadequate context, incorrect objective, broken differentiation or missing update. Include occasional useful small experiments and reasoning pauses with hints/verification.

Close by reconstructing the decisions taken and the next missing piece. Avoid routine interview lists, MCQs and exam answers. Student mastery remains pending until demonstrated.

## Approved scope and location

The user approved the sample on 2026-10-15. Apply its concept-and-reasoning flow to all current Phase 1 chapters/companions and future teaching. Approval is granted; do not repeat the sample gate. Preserve the approved sample as a reference and archive old content before revisions.

Canonical lessons remain in C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes. Use absolute source/report links and relative companion links. Record references and hashes in the code repository, never recreate its notes directory. Archive previous content before a later approved rewrite.

## Completed Phases 5–6 and loss-teaching requirement — 2026-10-18

Phases 5 and 6 each have six natural implementation commits and separate Desktop phase05/phase06 concept chapters, companions, reconstructions and LOSS_AND_LEARNING.md. Read reports/phase05/SUMMARY.md and reports/phase06/SUMMARY.md before Phase 7. Tiny-batch NLL 6.242738 → 0.001790 is memorization of 124 repeated events. The separate ten-update CPU pilot reduces full development NLL 6.246756 → 5.843923 on 39,711 targets. CPU fresh-process full-state resume matches its recorded reference. The 6,605,312-parameter CUDA candidate is a resource pilot; main training remains Phase 7. Preserve earlier Phase 1–4 chapters and the final holdout. Learner understanding remains pending.

The user explicitly requires teaching how and why loss changes in relation to what we actually did. Carry this into every future phase: derive the metric, connect gradients/updates and data exposure to its movement, state population/mode/timing/mask denominator, use actual charts where useful, explain generalization and confounds, and diagnose realistic alternatives. Separate observed changes from hypotheses. Never invent intermediate curves or claim an individual component caused a combined improvement without a controlled measurement.
