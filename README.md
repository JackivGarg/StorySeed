# StorySeed

A small language model built and trained from random initialization for short English story continuation.

This fresh repository began on 2026-10-14. Its Phase 1 code and lessons were developed here without importing earlier project history or source files. Phase 1 now trains, generates and saves/reloads a small character bigram model; the final transformer is a later phase.

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
