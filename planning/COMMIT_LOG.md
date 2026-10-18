# Actual StorySeed history

| Commit | Change |
|---|---|
| `a332e7a` | chore: initialize fresh StorySeed project |
| `f676586` | docs: establish StorySeed learning and commit workflow |
| `b703dda` | docs: record successful initial GitHub publication |
| `5773fd9` | chore: establish the PyTorch environment and inspect gradients |
| `43da6dd` | feat: turn toy stories into inspectable next-token examples |
| `322c2c4` | feat: assemble aligned tensor windows and repeatable batches |
| `84a001a` | feat: implement a trainable bigram score table |
| `d1cb25d` | feat: measure next-token error and inspect cross-entropy gradients |
| `3377bd4` | feat: train the character bigram model from random weights |
| `e60b4db` | feat: sample bounded text continuations from learned scores |
| `5326d0c` | feat: save and reload the trained model in a fresh process |

The current change cannot contain its own hash; use git log to locate it. Add its known hash during the next real change.

## Phase 2 implementation commits

| Stage | Commit | Change |
|---|---|---|
| 2.1 | 25609f8 | Corpus provenance |
| 2.2 | 39ce747 | Whole-story splits |
| 2.3 | 532f7d9 | Explicit BPE |
| 2.4 | bd8c46b | Frozen tokenizer |
| 2.5 | e62fe7e | Boundary-aware batches |
| 2.6 | 1a39605 | Comparable trained baseline |

## Phase 3 implementation commits

| Stage | Commit | Change |
|---|---|---|
| 3.1 | b700b50 | Shared trainable features |
| 3.2 | a0fc359 | Ordered contextual predictor |
| 3.3 setup | 737ae66 | Predeclared training protocol |
| 3.3 outcome | 5ed57b1 | Actual learning and baseline gap |
| 3.4 | effd9d4 | Causal generation and information boundary |

## Phase 4 implementation commits

| Stage | Commit | Change |
|---|---|---|
| 4.1 | 0750d9a | Retrieval-role projections |
| 4.2 | 8b727f4 | Scaled position matches and values |
| 4.3 | 4e0e7ea | Causal/padding restrictions |
| 4.4 | 430d0b6 | Learned window positions |
| 4.5 setup | 3489023 | Integrated model and declared pilot |
| 4.5 outcome | 0a436a4 | Actual learning, information flow and restored inference |

## Phase 5 implementation commits

| Stage | Commit | Change |
|---|---|---|
| 5.1 | 5283713 | Multiple causal retrieval paths |
| 5.2 | 651cf3d | Nonlinear position-wise processing |
| 5.3 | 4fea6b9 | Residual paths and pre-normalization |
| 5.4 | 64e1c33 | Explicit regularization and modes |
| 5.5 | edfb7f6 | Complete stack and parameter ledger |
| 5.6 | 5dff654 | Actual tiny fit, causal observations and inference |

## Phase 6 implementation commits

| Stage | Commit | Change |
|---|---|---|
| 6.1 | 556bfb8 | Training/evaluation contract |
| 6.2 | 120000f | AdamW groups, clipping and schedule |
| 6.3 | 076737e | Global true-target accumulation |
| 6.4 | 5fc1247 | Actual CUDA precision and scaling |
| 6.5 | e2c5e72 | Full-state fresh-process resume |
| 6.6 | ee6abb8 | Resource candidate and end-to-end public pilot |

# Phase 7: preparation commits and evidence

These natural increments implement the upcoming workflow. Setup is distinct from completed execution; actual result commits will follow the job.

| Stage | Commit | Chapter | Setup record |
|---|---|---|---|
| 7.1 | [e6929f5](https://github.com/JackivGarg/StorySeed/commit/e6929f5) | [01-decide-what-a-main-run-means.md](01-decide-what-a-main-run-means.md) | reports/phase07/01_setup.json |
| 7.2 | [c781c22](https://github.com/JackivGarg/StorySeed/commit/c781c22) | [02-follow-the-run-through-successful-updates.md](02-follow-the-run-through-successful-updates.md) | reports/phase07/02_setup.json |
| 7.3 | [79d6545](https://github.com/JackivGarg/StorySeed/commit/79d6545) | [03-connect-curves-to-what-the-model-learned.md](03-connect-curves-to-what-the-model-learned.md) | reports/phase07/03_setup.json |
| 7.4 | [b51e397](https://github.com/JackivGarg/StorySeed/commit/b51e397) | [04-make-the-learned-model-usable-in-another-process.md](04-make-the-learned-model-usable-in-another-process.md) | reports/phase07/04_setup.json |

Source/configuration and small reports are in the repository. Detailed chapters remain in this separate Desktop folder. The upcoming run writes observed records; a setup commit is not proof of a trained model or final score.

# Phase 8: preparation commits and evidence

These natural increments implement the upcoming workflow. Setup is distinct from completed execution; actual result commits will follow the job.

| Stage | Commit | Chapter | Setup record |
|---|---|---|---|
| 8.1 | [13fb453](https://github.com/JackivGarg/StorySeed/commit/13fb453) | [01-turn-a-working-model-into-a-research-question.md](01-turn-a-working-model-into-a-research-question.md) | reports/phase08/01_setup.json |
| 8.2 | [1502227](https://github.com/JackivGarg/StorySeed/commit/1502227) | [02-measure-prediction-without-changing-the-question.md](02-measure-prediction-without-changing-the-question.md) | reports/phase08/02_setup.json |
| 8.3 | [7df1dde](https://github.com/JackivGarg/StorySeed/commit/7df1dde) | [03-read-paired-stories-without-selecting-the-nicest-example.md](03-read-paired-stories-without-selecting-the-nicest-example.md) | reports/phase08/03_setup.json |
| 8.4 | [bea8344](https://github.com/JackivGarg/StorySeed/commit/bea8344) | [04-compare-contexts-and-understand-uncertainty.md](04-compare-contexts-and-understand-uncertainty.md) | reports/phase08/04_setup.json |
| 8.5 | [95ef17a](https://github.com/JackivGarg/StorySeed/commit/95ef17a) | [05-write-the-paper-from-the-evidence.md](05-write-the-paper-from-the-evidence.md) | reports/phase08/05_setup.json |

Source/configuration and small reports are in the repository. Detailed chapters remain in this separate Desktop folder. The upcoming run writes observed records; a setup commit is not proof of a trained model or final score.

## Phases 7–8 actual outcome commits

- 7.2: a3bb732 — reports/phase07/main128.json, reports/phase07/context64.json, reports/phase07/SUMMARY.md
- 7.3: 587e7a4 — reports/phase07/main128-curve.json, reports/phase07/context64-curve.json, reports/phase07/main128-loss-interpretation.json, reports/phase07/context64-loss-interpretation.json
- 7.4: ebfa7b8 — reports/phase07/04_fresh_inference.json
- 8.1: 00312ff — data/manifests/phase08/final.json
- 8.2: dfd9885 — reports/phase08/metrics.json
- 8.3: 6b4a573 — reports/phase08/samples.json, reports/phase08/PAIRED_SAMPLE_REVIEW.md, reports/phase08/review_key.json, reports/phase08/review_ratings.csv, reports/phase08/ASSISTANT_SAMPLE_REVIEW.md, src/storyseed/sample_review.py
- 8.4: 033f272 — reports/phase08/CONTEXT_STUDY.md, reports/phase08/matched_resources.json
- 8.5: f849497 — reports/phase08/PAPER_DRAFT.md, reports/phase08/SUMMARY.md, reports/phase08/chart_assets.json

Phase 7 intermediate evidence: capture `34ef116`; early learning `9159ffa`; recovery `08ba38e`; incident/device observations `cba39a9`. These precede the recorded final outcome commits.

## Phase 9 focused commits

- f1f231d feat: give the trained main model a consistent generation command
- 467c840 feat: add a responsive local story continuation window
- 2827a50 feat: package the fixed model and tokenizer for local use
- 469182b docs: explain the model and university evidence honestly
- 4b50ae0 docs: connect the complete build to independent reconstruction

Final closure and release records follow in later commits. Source and notes are delivered; independent learning evidence remains pending.
