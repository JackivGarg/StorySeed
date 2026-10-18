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

## Phase 3.2

Chapter: phase03/02-build-an-ordered-context-predictor.md. Companion: phase03/02-thinking-and-inspection.md. Code: src/storyseed/fixed_context.py. Actual evidence: reports/phase03/02_context.json. Written, implemented, executed; learner understanding pending.

## Phase 3.3

Chapter: phase03/03-make-context-features-learn.md. Companion: phase03/03-thinking-and-inspection.md. Code: src/storyseed/context_training.py. Actual evidence: reports/phase03/03_training_k8.json. Written, implemented, executed; learner understanding pending.

## Phase 3.4

Chapter: phase03/04-discover-what-fixed-context-cannot-do.md. Companion: phase03/04-thinking-and-inspection.md. Code: src/storyseed/context_generation.py. Actual evidence: reports/phase03/04_limits.json. Written, implemented, executed; learner understanding pending.

## Phase 3 implementation commits

| Stage | Commit | Change |
|---|---|---|
| 3.1 | b700b50 | Shared trainable features |
| 3.2 | a0fc359 | Ordered contextual predictor |
| 3.3 setup | 737ae66 | Predeclared training protocol |
| 3.3 outcome | 5ed57b1 | Actual learning and baseline gap |
| 3.4 | effd9d4 | Causal generation and information boundary |

## Phase 4.1

Chapter: phase04/01-learn-to-request-and-offer-information.md; companion: phase04/01-thinking-and-inspection.md. Code: src/storyseed/attention.py. Actual evidence: reports/phase04/01_qkv.json. Written, implemented and executed; learner understanding pending.

## Phase 4.2

Chapter: phase04/02-turn-matches-into-retrieved-information.md; companion: phase04/02-thinking-and-inspection.md. Code: src/storyseed/attention.py. Actual evidence: reports/phase04/02_retrieval.json. Written, implemented and executed; learner understanding pending.

## Phase 4.3

Chapter: phase04/03-make-the-past-the-only-available-source.md; companion: phase04/03-thinking-and-inspection.md. Code: src/storyseed/attention.py. Actual evidence: reports/phase04/03_causality.json. Written, implemented and executed; learner understanding pending.

## Phase 4.4

Chapter: phase04/04-give-retrieval-a-sense-of-position.md; companion: phase04/04-thinking-and-inspection.md. Code: src/storyseed/position_features.py. Actual evidence: reports/phase04/04_positions.json. Written, implemented and executed; learner understanding pending.

## Phase 4.5

Chapter: phase04/05-build-and-train-the-attention-predictor.md; companion: phase04/05-thinking-and-inspection.md. Code: src/storyseed/attention_model.py. Actual evidence: reports/phase04/05_training.json. Written, implemented and executed; learner understanding pending.

## Phase 4 implementation commits

| Stage | Commit | Change |
|---|---|---|
| 4.1 | 0750d9a | Retrieval-role projections |
| 4.2 | 8b727f4 | Scaled position matches and values |
| 4.3 | 4e0e7ea | Causal/padding restrictions |
| 4.4 | 430d0b6 | Learned window positions |
| 4.5 setup | 3489023 | Integrated model and declared pilot |
| 4.5 outcome | 0a436a4 | Actual learning, information flow and restored inference |

## Phase 5.1

Read phase05/01-preserve-several-retrieval-paths.md, then 01-thinking-and-inspection.md in that folder. Code: src/storyseed/decoder_components.py, src/storyseed/build_evidence.py, scripts/phase05/01_inspect_heads.py. Actual evidence: reports/phase05/01_heads.json. Written, implemented and executed; student understanding pending.

## Phase 5.2

Read phase05/02-process-the-retrieved-features.md, then 02-thinking-and-inspection.md in that folder. Code: src/storyseed/decoder_components.py, scripts/phase05/02_inspect_feedforward.py. Actual evidence: reports/phase05/02_feedforward.json. Written, implemented and executed; student understanding pending.

## Phase 5.3

Read phase05/03-keep-a-trainable-path-through-the-block.md, then 03-thinking-and-inspection.md in that folder. Code: src/storyseed/decoder_components.py, scripts/phase05/03_inspect_residuals.py. Actual evidence: reports/phase05/03_residuals.json. Written, implemented and executed; student understanding pending.

## Phase 5.4

Read phase05/04-understand-training-and-evaluation-modes.md, then 04-thinking-and-inspection.md in that folder. Code: src/storyseed/decoder_components.py, scripts/phase05/04_inspect_modes.py. Actual evidence: reports/phase05/04_modes.json. Written, implemented and executed; student understanding pending.

## Phase 5.5

Read phase05/05-assemble-the-complete-decoder.md, then 05-thinking-and-inspection.md in that folder. Code: src/storyseed/decoder.py, scripts/phase05/05_inspect_stack.py. Actual evidence: reports/phase05/05_stack.json. Written, implemented and executed; student understanding pending.

## Phase 5.6

Read phase05/06-make-the-whole-decoder-learn-a-tiny-batch.md, then 06-thinking-and-inspection.md in that folder. Code: src/storyseed/decoder_io.py, scripts/generate_decoder.py, scripts/phase05/06_learn_tiny_batch.py, data/manifests/phase05/tiny-decoder.json. Actual evidence: reports/phase05/06_tiny_learning.json. Written, implemented and executed; student understanding pending.

## Phase 6.1

Read phase06/01-define-the-run-before-updating-weights.md, then 01-thinking-and-inspection.md in that folder. Code: src/storyseed/training_data.py, src/storyseed/training_config.py, src/storyseed/training_evaluation.py, configs/phase06_teaching.json, scripts/phase06/01_define_run.py. Actual evidence: reports/phase06/01_run_contract.json. Written, implemented and executed; student understanding pending.

## Phase 6.2

Read phase06/02-control-the-optimizer-update.md, then 02-thinking-and-inspection.md in that folder. Code: src/storyseed/training_optimizer.py, scripts/phase06/02_inspect_optimizer.py. Actual evidence: reports/phase06/02_optimizer.json. Written, implemented and executed; student understanding pending.

## Phase 6.3

Read phase06/03-accumulate-the-right-token-objective.md, then 03-thinking-and-inspection.md in that folder. Code: src/storyseed/training_runner.py, scripts/phase06/03_inspect_accumulation.py. Actual evidence: reports/phase06/03_accumulation.json. Written, implemented and executed; student understanding pending.

## Phase 6.4

Read phase06/04-use-precision-that-the-machine-can-support.md, then 04-thinking-and-inspection.md in that folder. Code: src/storyseed/training_runner.py, scripts/phase06/04_inspect_precision.py. Actual evidence: reports/phase06/04_precision.json. Written, implemented and executed; student understanding pending.

## Phase 6.5

Read phase06/05-resume-the-same-training-process.md, then 05-thinking-and-inspection.md in that folder. Code: src/storyseed/training_runner.py, src/storyseed/training_checkpoint.py, scripts/train_decoder.py, scripts/phase06/05_inspect_resume.py, data/manifests/phase06/resume-checkpoint.json. Actual evidence: reports/phase06/05_resume.json. Written, implemented and executed; student understanding pending.

## Phase 6.6

Read phase06/06-measure-a-feasible-training-budget.md, then 06-thinking-and-inspection.md in that folder. Code: configs/phase06_candidate.json, scripts/phase06/06_measure_resources.py, scripts/phase06/07_record_pipeline.py, reports/phase06/06_resources_initial.json, reports/phase06/06_resources_initial.md, reports/phase06/07_pipeline.json, reports/phase06/07_pipeline.md, data/manifests/phase06/pilot-artifacts.json. Actual evidence: reports/phase06/06_resources.json. Written, implemented and executed; student understanding pending.

## Phase 5 implementation commits

| Stage | Commit | Change |
|---|---|---|
| 5.1 | 5283713 | Multiple causal retrieval paths |
| 5.2 | 651cf3d | Nonlinear position-wise processing |
| 5.3 | 4fea6b9 | Residual paths and pre-normalization |
| 5.4 | 64e1c33 | Explicit regularization and modes |
| 5.5 | edfb7f6 | Complete stack and parameter ledger |
| 5.6 | 5dff654 | Actual tiny fit, causal observations and inference |

External phase05/README.md gives the concept order; LOSS_AND_LEARNING.md explains actual metric changes and charts. The external IMPLEMENTATION_AND_EVIDENCE.md links each hash to its source and report.

## Phase 6 implementation commits

| Stage | Commit | Change |
|---|---|---|
| 6.1 | 556bfb8 | Training/evaluation contract |
| 6.2 | 120000f | AdamW groups, clipping and schedule |
| 6.3 | 076737e | Global true-target accumulation |
| 6.4 | 5fc1247 | Actual CUDA precision and scaling |
| 6.5 | e2c5e72 | Full-state fresh-process resume |
| 6.6 | ee6abb8 | Resource candidate and end-to-end public pilot |

External phase06/README.md gives the concept order; LOSS_AND_LEARNING.md explains actual metric changes and charts. The external IMPLEMENTATION_AND_EVIDENCE.md links each hash to its source and report.

## Phase 7.1 preparation — 2026-10-18

Read external phase07/01-decide-what-a-main-run-means.md, then 01-thinking-and-inspection.md. Sources: .gitattributes, configs/phase07_main.json, planning/PHASE07_08_PROTOCOL.json, src/storyseed/experiment_data.py. Setup record: reports/phase07/01_setup.json. Notes/source prepared; actual long execution pending, not claimed complete.

## Phase 7.2 preparation — 2026-10-18

Read external phase07/02-follow-the-run-through-successful-updates.md, then 02-thinking-and-inspection.md. Sources: src/storyseed/training_runner.py, src/storyseed/training_checkpoint.py, src/storyseed/experiment_training.py, scripts/train_experiment_condition.py. Setup record: reports/phase07/02_setup.json. Notes/source prepared; actual long execution pending, not claimed complete.

## Phase 7.3 preparation — 2026-10-18

Read external phase07/03-connect-curves-to-what-the-model-learned.md, then 03-thinking-and-inspection.md. Sources: scripts/describe_training_curve.py. Setup record: reports/phase07/03_setup.json. Notes/source prepared; actual long execution pending, not claimed complete.

## Phase 7.4 preparation — 2026-10-18

Read external phase07/04-make-the-learned-model-usable-in-another-process.md, then 04-thinking-and-inspection.md. Sources: scripts/verify_main_export.py. Setup record: reports/phase07/04_setup.json. Notes/source prepared; actual long execution pending, not claimed complete.

## Phase 8.1 preparation — 2026-10-18

Read external phase08/01-turn-a-working-model-into-a-research-question.md, then 01-thinking-and-inspection.md. Sources: planning/PHASE07_08_FREEZE.json, planning/PHASE08_REVIEW_RUBRIC.md. Setup record: reports/phase08/01_setup.json. Notes/source prepared; actual long execution pending, not claimed complete.

## Phase 8.2 preparation — 2026-10-18

Read external phase08/02-measure-prediction-without-changing-the-question.md, then 02-thinking-and-inspection.md. Sources: src/storyseed/research_metrics.py. Setup record: reports/phase08/02_setup.json. Notes/source prepared; actual long execution pending, not claimed complete.

## Phase 8.3 preparation — 2026-10-18

Read external phase08/03-read-paired-stories-without-selecting-the-nicest-example.md, then 03-thinking-and-inspection.md. Sources: src/storyseed/sample_review.py. Setup record: reports/phase08/03_setup.json. Notes/source prepared; actual long execution pending, not claimed complete.

## Phase 8.4 preparation — 2026-10-18

Read external phase08/04-compare-contexts-and-understand-uncertainty.md, then 04-thinking-and-inspection.md. Sources: src/storyseed/context_comparison.py, src/storyseed/research_evaluation.py. Setup record: reports/phase08/04_setup.json. Notes/source prepared; actual long execution pending, not claimed complete.

## Phase 8.5 preparation — 2026-10-18

Read external phase08/05-write-the-paper-from-the-evidence.md, then 05-thinking-and-inspection.md. Sources: scripts/summarize_phase07_08.py, scripts/run_phase07_08.py. Setup record: reports/phase08/05_setup.json. Notes/source prepared; actual long execution pending, not claimed complete.

Phase 7.1 observed capture: 7,585 training stories / 2,818,569 available targets; full initialization development NLL 6.295087 over 39,711 events. Read external phase07/01-observed-data-and-the-first-loss.md. Main training remains active.

Phase 7.2 actual outcome: reports/phase07/main128.json, reports/phase07/context64.json, reports/phase07/SUMMARY.md. Read separate phase07/ACTUAL_LOSS_AND_DECISIONS.md and OBSERVED_RESULTS.md after its concept chapter. Human review/learner understanding remain pending.

Phase 7.3 actual outcome: reports/phase07/main128-curve.json, reports/phase07/context64-curve.json, reports/phase07/main128-loss-interpretation.json, reports/phase07/context64-loss-interpretation.json. Read separate phase07/ACTUAL_LOSS_AND_DECISIONS.md and OBSERVED_RESULTS.md after its concept chapter. Human review/learner understanding remain pending.

Phase 7.4 actual outcome: reports/phase07/04_fresh_inference.json. Read separate phase07/ACTUAL_LOSS_AND_DECISIONS.md and OBSERVED_RESULTS.md after its concept chapter. Human review/learner understanding remain pending.

Phase 8.1 actual outcome: data/manifests/phase08/final.json. Read separate phase08/ACTUAL_LOSS_AND_DECISIONS.md and OBSERVED_RESULTS.md after its concept chapter. Human review/learner understanding remain pending.

Phase 8.2 actual outcome: reports/phase08/metrics.json. Read separate phase08/ACTUAL_LOSS_AND_DECISIONS.md and OBSERVED_RESULTS.md after its concept chapter. Human review/learner understanding remain pending.

Phase 8.3 actual outcome: reports/phase08/samples.json, reports/phase08/PAIRED_SAMPLE_REVIEW.md, reports/phase08/review_key.json, reports/phase08/review_ratings.csv, reports/phase08/ASSISTANT_SAMPLE_REVIEW.md. Read separate phase08/ACTUAL_LOSS_AND_DECISIONS.md and OBSERVED_RESULTS.md after its concept chapter. Human review/learner understanding remain pending.

Phase 8.3 actual outcome: reports/phase08/samples.json, reports/phase08/PAIRED_SAMPLE_REVIEW.md, reports/phase08/review_key.json, reports/phase08/review_ratings.csv, reports/phase08/ASSISTANT_SAMPLE_REVIEW.md, src/storyseed/sample_review.py. Read separate phase08/ACTUAL_LOSS_AND_DECISIONS.md and OBSERVED_RESULTS.md after its concept chapter. Human review/learner understanding remain pending.
