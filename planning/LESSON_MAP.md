# Desktop lesson map

Authoritative notes folder: `C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes`.
Code folder: `C:/Users/JACKIV/OneDrive/Desktop/StorySeed`.

Open `START_HERE.md` in the notes folder. The notes were initially copied unchanged, then rewritten on 2026-10-14 at the user's explicit request as practical guided build sessions. Notes are stored locally outside this Git repository. Code, execution reports and operational planning remain here.

## Approved conceptual teaching — 2026-10-15

The user approved the sample and requested a full rewrite of current teaching. All Phase 1 chapters and reasoning companions now follow the conceptual build journey. Begin at phase01/README.md. Extra guides: phase01/01-working-with-the-project-environment.md, phase01/COVERAGE.md and THE_SLM_JOURNEY.md. The approved sample is retained in samples.

The previous current notes are archived in history/b7ebfb7-before-approved-concept-flow. Existing earlier archives are preserved. The listed code commits remain the original implementation increments, not invented new code changes for this teaching revision.

## Phase 1

All chapter paths below are relative to the external notes folder. Each session has a companion hands-on lab `phase01/NN-practice-and-solutions.md` with the same two-digit prefix. Filenames were preserved for continuity; these are practical labs, not exam answers. The current rewritten lessons explain the existing implementation, while the listed commits identify its original build increments.

| Subphase | Code commit | Chapter |
|---|---|---|
| 1.1 | 5773fd9 | phase01/01-environment-tensors-and-gradients.md |
| 1.2 | 43da6dd | phase01/02-text-tokens-and-targets.md |
| 1.3 | 322c2c4 | phase01/03-windows-batches-and-shapes.md |
| 1.4 | 84a001a | phase01/04-bigram-scores.md |
| 1.5 | d1cb25d | phase01/05-probability-and-cross-entropy.md |
| 1.6 | 3377bd4 | phase01/06-training-and-updates.md |
| 1.7 | e60b4db | phase01/07-generation-and-sampling.md |
| 1.8 | 5326d0c | phase01/08-saving-and-loading.md |

The phase build guide is `phase01/README.md`; the whole-system trace is `phase01/00-follow-one-character-through-the-system.md`. `READING_ORDER.md` links sessions, labs, actual source and reports. Rewritten notes use absolute source/report links and relative companion links, appropriate to their separate folder.

The original pre-rewrite notes and indexes remain in `history/059c13b-before-practical-rewrite/`. That archive identifies the code revision before the style rewrite. Historical Phase 1 notes also remain accessible through `git show <commit>:notes/phase01/<filename>`. The current external note hashes are recorded in `planning/LESSON_MANIFEST.json` for detecting later edits; this manifest is not a substitute for the note content or its external archive.

## Future subphases

Continue writing detailed chapters and practice in the external folder. Update this map alongside each natural code commit. Before revising an existing chapter, preserve its prior content in the external folder under `history/<code-commit>/`. Keep understanding checkpoints pending until supported by student evidence.

## Phase 2 — story pipeline

External folder: phase02. The learner requested this phase on 2026-10-16 with the approved concept-and-flow style. Phase 1 notes remain unchanged.

| Subphase | Chapter | Code | Status |
|---|---|---|---|
| 2.1 | phase02/01-why-the-corpus-is-part-of-the-model.md | src/storyseed/corpus.py | Written, implemented and executed; understanding pending |
| 2.2 | phase02/02-why-the-split-comes-before-learning.md | src/storyseed/story_splits.py | Written, implemented and executed; understanding pending |
| 2.3 | phase02/03-learn-byte-pair-merging.md | src/storyseed/bpe_reference.py | Written, implemented and executed; understanding pending |
| 2.4 | phase02/04-choose-and-freeze-the-tokenizer.md | src/storyseed/story_tokenizer.py | Written, implemented and executed; understanding pending |
| 2.5 | phase02/05-preserve-the-task-while-batching.md | src/storyseed/story_batches.py | Written, implemented and executed; understanding pending |
| 2.6 | phase02/06-establish-an-honest-baseline.md | src/storyseed/story_baseline.py | Written, implemented and executed; understanding pending |

## Phase 2 implementation commits

| Stage | Commit | Change |
|---|---|---|
| 2.1 | 25609f8 | Corpus provenance |
| 2.2 | 39ce747 | Whole-story splits |
| 2.3 | 532f7d9 | Explicit BPE |
| 2.4 | bd8c46b | Frozen tokenizer |
| 2.5 | e62fe7e | Boundary-aware batches |
| 2.6 | 1a39605 | Comparable trained baseline |

## Phase 3.1

Chapter: phase03/01-why-prediction-needs-learned-features.md. Companion: phase03/01-thinking-and-inspection.md. Code: src/storyseed/token_features.py. Actual evidence: reports/phase03/01_embeddings.json. Written, implemented, executed; learner understanding pending.
