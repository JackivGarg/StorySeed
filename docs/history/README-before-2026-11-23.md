# StorySeed

## Quality recovery completed — 2026-10-21

The separate recovery completed 48,000 random-initialized updates. Its development-selected model is StorySeed-recovery4096: 8,473,088 parameters, 4,096-entry BPE and 256-ID context. The current app now loads that declared artifact rather than the old 512-entry prototype. Restart any already-open app to load the update.

Same-text final canonical encoded-path bits per UTF-8 content byte: **1.707953 → 0.626487**, across 200 newly reserved stories after candidate freeze. Read reports/quality_recovery/ASSISTANT_REVIEW.md for the identified review of all 16 sample pairs and remaining failures. This is a combined recovery, not an isolated ablation or independent human study. Both final populations are now used evidence.

Word formation and readability improved substantially. Reliable story continuation remains unachieved: recurring reference confusion and weak plot logic remain. The current release is an improved experimental prototype, not a claim that the broader quality goal is complete.

Actual integration evidence covers CPU generation, a withdrawn Tk submit/display/save callback and fresh isolated-process source/weight relocation on this laptop. The Save path was supplied by the observation harness; human visual review and a clean second-machine dependency installation are unclaimed. Learner reconstruction and independent human ratings remain pending. No training job remains running.

Start reading the separate Desktop StorySeed-Study-Notes/quality-recovery/README.md. Current setup/use: docs/LOCAL_APP.md. Model and university evidence: docs/RECOVERY_MODEL_CARD.md and docs/QUALITY_RECOVERY_REPORT.md. Portable release identity: reports/quality_recovery/FINAL_RELEASE.json. Earlier phase results and chapters remain historical evidence.

## Earlier progression

## Active quality recovery — continuation running

The original app's poor story quality is not treated as solved. A separate scratch recovery completed 16,000 updates, but its diagnostic stories still confused characters. An additional 32,000-update full-state stage is running under planning/QUALITY_RECOVERY_CONTINUATION_PROTOCOL.json. Current counters and checkpoint: reports/quality_recovery/LIVE_STATUS.json. Native execution session: 71643 in the active chat; do not launch a duplicate.

The new final stories remain unencoded/unscored; evaluation and application promotion are pending. The current app still serves the historical model. Read the separate Desktop StorySeed-Study-Notes/quality-recovery folder for diagnosis, actual learning and the continuation decision. Earlier pipeline completion is not a quality guarantee.

## Earlier progression

## Use the trained local app

```powershell
.venv/Scripts/python -m storyseed.app --gui
.venv/Scripts/python -m storyseed.app --prompt "Once upon a time" --json
```

Use [local setup and release instructions](docs/LOCAL_APP.md), [model card](docs/MODEL_CARD.md), [university report](docs/UNIVERSITY_REPORT.md), [demo guide](docs/DEMO_GUIDE.md) and [independent reconstruction workbook](docs/RECONSTRUCTION_WORKBOOK.md). Trained weights remain local/ignored: cloning this repository alone does not supply them. Final complete release identity: reports/phase09/FINAL_RELEASE.json.

All nine planned implementation phases are delivered, with detailed concept-first notes in the separate Desktop StorySeed-Study-Notes/phase09 folder. The app serves the predeclared main128 model, not newly tuned weights. Grammar and continuity limitations remain. Human ratings and independent student reconstruction are pending. Older progress entries below preserve historical states.

## Historical progression

A small language model built and trained from random initialization for short English story continuation.

This fresh repository began on 2026-10-14. Its Phase 1 code and lessons were developed here without importing earlier project history or source files. Phase 1 trained and saved a small character bigram; Phases 2–6 now add the story pipeline, contextual baselines, explicit complete causal decoder and repeatable training system. Main model training and fixed evaluation now run in completed Phases 7–8; Phase 9 now supplies the local application, portable release and reconstruction guidance.

Learning proceeds alongside implementation through detailed, connected chapters. The intended outcome is understanding tokenization, the causal transformer, training, evaluation, and generation.

The active teaching approach follows the concepts and reasoning needed to build each component: project need, prerequisite learning, return to the build, implementation and the next limitation. The user approved the PyTorch/learning-mechanism sample on 2026-10-15. All current Phase 1 teaching files now follow that style in the separate Desktop notes folder. See [the approved teaching reference](planning/TEACHING_PREVIEW.md) and [teaching agreement](planning/TEACHING_RULES.md).

## Start here

- [Build-and-teach plan: 49 core subphases](planning/END_TO_END_PLAN.md).
- [Detailed notes standard](planning/NOTES_STANDARD.md).
- [Commit-based learning workflow](planning/COMMIT_WORKFLOW.md).
- [Study book location and chapter map](planning/LESSON_MAP.md).
- [Current progress](planning/PROGRESS.md).

Each subphase pairs a natural code commit with its explanation and observed behavior. Main weights start randomly. Model size and training budget follow a measured hardware pilot.

Branch: main. Remote origin: https://github.com/JackivGarg/StorySeed.git. Git authentication and publication succeeded. Earlier project folders are preserved as references.

## Phase 1: run and study

Teaching notes are stored separately on this computer in `C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes`. Open `START_HERE.md` there for the study guide and reading index. See [the chapter map](planning/LESSON_MAP.md) for corresponding code commits. The original eight subphase commits still contain their historical explanations; current lessons live in the separate Desktop folder.

On this machine the project environment reuses the installed PyTorch CUDA build through system-site-packages. The environment and a fully isolated alternate installation are explained in the first chapter. After setup:

```powershell
.venv\Scripts\python scripts\train_phase01.py --device auto
.venv\Scripts\python scripts\generate.py --prompt "mia " --seed 17
.venv\Scripts\python scripts\generate.py --prompt "mia " --top-k 1
.venv\Scripts\python -m unittest discover -s tests -v
```

Training starts with random weights. The local checkpoint is checkpoints/phase01/bigram.pt and stays outside Git. Recreate it using the first command; its observed artifact identity is recorded in reports/phase01/08_checkpoint.json. Generation loads weights without retraining. CPU training/generation also ran successfully.

The toy model has 289 parameters and was trained on 16 authored sentences. CUDA training loss fell from 2.835582 to 1.212152; development loss fell from 2.834067 to 1.216261 over 400 steps. Its one-character context can learn transitions but cannot reliably write coherent stories. All eight contract checks passed. Implementation completeness does not establish student mastery; learning checkpoints remain pending.

## Phase 2: reproducible story pipeline

Six completed increments capture/split a pinned TinyStories teaching prefix, demonstrate BPE explicitly, fit/save a training-only tokenizer, build boundary-aware masked batches and train a compatible baseline. Read the external StorySeed-Study-Notes/phase02/README.md; [lesson references and original commits](planning/LESSON_MAP.md) remain in this repository.

Generate from its local checkpoint with `.venv/Scripts/python scripts/generate_phase02.py --json`. The final holdout remains reserved. See [the measured Phase 2 summary](reports/phase02/SUMMARY.md).

## Phase 3: learned representations and contextual prediction

The fixed-window neural baseline and its four build-and-teach subphases are implemented. Both K=1/8 conditions learned from random initialization, but this short pilot did not beat the Phase 2 bigram. See [actual outcome](reports/phase03/SUMMARY.md). The separate Desktop StorySeed-Study-Notes/phase03 folder contains the conceptual reading journey. Generate locally without fitting again:

```powershell
.venv/Scripts/python scripts/generate_phase03.py --prompt "Once upon a time" --json
```

The local checkpoint is ignored by Git; its identity is recorded in data/manifests/phase03. Attention and the final Transformer remain later phases.

## Phase 4: explicit causal attention

Q/K/V retrieval, score scaling, causal/padding restrictions and learned positions now form a trained single-head teaching predictor. See [actual outcomes and limitations](reports/phase04/SUMMARY.md). Detailed concept-first chapters are in the separate Desktop StorySeed-Study-Notes/phase04 folder. Generate without fitting again:

```powershell
.venv/Scripts/python scripts/generate_phase04.py --prompt "Once upon a time" --json
```

The local inference checkpoint remains ignored by Git and is identified in data/manifests/phase04/attention.json. This short pilot did not beat the previous baselines; the full decoder and main scratch-training run are later phases.

## Phases 5–6: complete decoder and dependable training

Six increments per phase implement the explicit pre-normalized decoder and a configurable training workflow with masked target accounting, AdamW/schedule/clipping, correct accumulation, measured CUDA precision and full-state resume. Read [Phase 5 results](reports/phase05/SUMMARY.md) and [Phase 6 results](reports/phase06/SUMMARY.md).

Detailed teaching is in the separate Desktop StorySeed-Study-Notes/phase05 and phase06 folders. Each has a loss-and-learning chapter with actual charts, explaining what changed and why the measured populations matter. The tiny memorization result is not a model-quality comparison. The ten-update CPU pilot improves full-development NLL but is not main training. The separate 6.6M CUDA model is only a measured resource candidate.

Generate from the local pilot export without retraining:

```powershell
.venv/Scripts/python scripts/generate_decoder.py --checkpoint checkpoints/phase06/pilot-inference.pt --json
```

The full training CLI is `scripts/train_decoder.py`; use `--help` and the recorded teaching config/protocol before starting a new run. Checkpoint tensors remain local/ignored. Next: Phase 7's frozen main-training protocol. Final holdout remains unencoded/unscored; student understanding remains pending.

## Phases 7–8: main training and research workflow

The separate Desktop phase07/phase08 notes teach the frozen expanded-corpus scratch run and context-policy study. Source and protocol are prepared; actual runtime status is in reports/phase07_08_STATUS.json. Main training and final metrics are not claimed before completion. The finite workload is scripts/run_phase07_08.py; do not launch a duplicate while its tracked job is active. Human review remains pending.

## Main scratch-trained result and evaluation

Phases 7–8 executed successfully. See reports/phase07/SUMMARY.md and reports/phase08/SUMMARY.md for actual measured results, resources, limits and artifact identities. The main local model is checkpoints/phase07/main128-inference.pt. The paired context study, generated samples and PAPER_DRAFT.md are in reports/phase08. Human ratings and learner understanding remain pending; Phase 9 is next.
