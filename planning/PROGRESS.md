# StorySeed progress

## Current phase — 2026-10-17

Phase 3: 4/4 subphases written, implemented and executed. The new external phase03 folder contains the connected concept journey and reasoning companions. K=1/8 both learn but do not beat the Phase 2 bigram; this is not the final Transformer. Phase 4 attention is next. Student understanding remains pending and the final holdout remains unencoded/unscored. Historical entries below preserve the earlier progression.

Active fresh repository: Desktop/StorySeed, branch main, origin https://github.com/JackivGarg/StorySeed.git. Earlier folders remain references.

Phase 1 implementation and its recorded verification are complete. Earlier teaching files exist, but the user requested a revised concept-and-flow style on 2026-10-15. The user approved the sample on 2026-10-15; all current Phase 1 teaching files now follow that conceptual approach. Understanding remains pending.

| Subphase | Build / actual run / notes | Student evidence |
|---|---|---|
| 1.1: environment | Implemented, executed, chapter and solutions written | Pending |
| 1.2: tokens | Implemented, executed, chapter and solutions written | Pending |
| 1.3: batches | Implemented, executed, chapter and solutions written | Pending |
| 1.4: bigram | Implemented, executed, chapter and solutions written | Pending |
| 1.5: loss | Implemented, executed, chapter and solutions written | Pending |
| 1.6: training | Implemented, executed, chapter and solutions written | Pending |
| 1.7: generation | Implemented, executed, chapter and solutions written | Pending |
| 1.8: checkpoint | Implemented, executed, chapter and solutions written | Pending |

Environment: project venv shares the installed PyTorch 2.6.0+cu124 dependency. CPU and CUDA gradient examples actually ran. Full dependency isolation is an alternate documented setup, not the environment used here.

Phase 1 uses an authored toy corpus and a character bigram model. It is not the final transformer or TinyStories training.

The user requested Phase 2 on 2026-10-16. Proceed with its connected explanations; prior and new student-understanding evidence remain pending. The new notes live in the separate phase02 folder.

Teaching update, 2026-10-14: the user authorized replacing the earlier teaching style with practical guided build sessions. All eight Phase 1 sessions and eight companion labs were rewritten against the actual code and reports. A complete data-flow guide connects the modules; reading indexes and permanent instructions now follow the mentor style. The originals were archived in the external notes folder at history/059c13b-before-practical-rewrite. Model source and historical execution reports were not changed for the rewrite. Student understanding evidence remains pending.

Phase 1 subphase commits and the final study-summary commit 2c7c874 were successfully pushed to origin/main on 2026-10-14. Git tracking state is authoritative for the latest publication status.

The user subsequently requested separating the notes and copying existing files unchanged. All 18 original note Markdown files (eight chapters, eight practice files, study guide and index) were copied byte for byte to C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes and removed from the current code tree. Code and execution reports remain here. Future lessons belong in that external folder; planning/LESSON_MAP.md links lessons to subphases. This relocation adds no model training or student-mastery evidence.

Actual results: 400 CUDA Adam updates, 289 parameters, 240 training/67 development prediction positions. Training loss 2.835582 to 1.212152; development loss 2.834067 to 1.216261. All eight contract tests passed. Saved weights and tokenizer reproduce the seeded result in a fresh process. CPU reproduction commands also ran. Local checkpoint: checkpoints/phase01/bigram.pt; weights remain ignored by Git, with a hash/manifest in reports/phase01/08_checkpoint.json.

## Phase 2 — requested next build

Completed increments: 6/6. Implementation and written teaching complete; Phase 3 is next. Student understanding is not inferred from assistant execution.

- 2.1 Traceable corpus: written, implemented, executed; student evidence pending.
- 2.2 Story splits and leakage: written, implemented, executed; student evidence pending.
- 2.3 How BPE discovers units: written, implemented, executed; student evidence pending.
- 2.4 A fitted, saved tokenizer: written, implemented, executed; student evidence pending.
- 2.5 Boundary-aware batches: written, implemented, executed; student evidence pending.
- 2.6 A comparable baseline: written, implemented, executed; student evidence pending.

Phase 2's six implementation commits through 1a39605 were successfully pushed to origin/main on 2026-10-16. The separate phase02 teaching folder contains 16 Markdown files; all current lesson hashes match the manifest and Phase 1 notes remain unchanged. Final closure documentation follows in its own commit; Git tracking state records its publication.

Phase 3.1: written, implemented and executed on 2026-10-17. See reports/phase03/01_embeddings.json and external phase03/01-why-prediction-needs-learned-features.md. Student evidence pending.

Phase 3.2: written, implemented and executed on 2026-10-17. See reports/phase03/02_context.json and external phase03/02-build-an-ordered-context-predictor.md. Student evidence pending.

Phase 3.3: written, implemented and executed on 2026-10-17. See reports/phase03/03_training_k8.json and external phase03/03-make-context-features-learn.md. Student evidence pending.

Phase 3.4: written, implemented and executed on 2026-10-17. See reports/phase03/04_limits.json and external phase03/04-discover-what-fixed-context-cannot-do.md. Student evidence pending.

Phase 3 implementation/protocol/result commits through effd9d4 were successfully pushed to origin/main on 2026-10-17. Final guide/closure documentation follows in its own commit; Git tracking state records its publication.
