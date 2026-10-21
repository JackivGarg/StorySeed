# Active recovery continuation: resume the actual work

Native execution session **71643** runs `.venv/Scripts/python scripts/continue_quality_recovery.py` in the active StorySeed repository. Initial session 15211 completed its 16,000 updates and exited successfully. Do not launch another training process merely because a tool session ID becomes unavailable. Inspect LIVE_STATUS.json, checkpoint timestamps and the actual process first.

The continuation adds 32,000 stage-local updates to the initial 16,000, for 48,000 overall. Full checkpoints are written every 250 stage updates to checkpoints/quality_recovery_continued/latest.pt; history is runs/quality_recovery_continued/history.json. New protocol, data identity and parent checkpoint hash are validated on restore. Only if the process is confirmed exited and no completed 03b_training.json exists, run the same continuation command to restore the saved boundary. Do not change frozen files casually.

No CPJ job ID, durable CPJ tracking or automatic completion notification exists on Windows. Keep the laptop plugged in and awake with its lid open, and keep Codex open for this native session. The finite keep-awake request does not override explicit lid/button sleep. The user authorizes several hours/overnight.

## Complete the remaining authorized work

1. Inspect actual 03b_training.json and 03b_curve.json after all 32,000 stage updates. Preserve initial 03_training.json/03_curve.json. Write the completed-continuation teaching chapter and natural result commit. The first 17,000 marker was slightly worse; no improvement is predetermined.
2. Only after the full declared extension completes, run scripts/evaluate_quality_recovery.py once. It requires the continued training record, freezes selected model/tokenizer/base and continuation protocol hashes, then encodes/scores the 200 new final stories. The old final set is not rescored. Once evaluated, the new set is used evidence.
3. Read ALL 16 pairs in the 32 raw outputs, not only 80-word display excerpts. Write identified ASSISTANT_REVIEW.md and truthful ASSISTANT_ASSESSMENT.json. Human ratings remain pending. If quality still fails, say so; do not promote by fabricating the required assessment.
4. Prepare actual figures/results/model card/university report, then commit the fixed evaluation and review separately. Promote only a supported candidate; observe actual CPU/Tk submit/display/save and inspect its text.
5. Commit clean integration source, package the new source/weight/tokenizer release, reproduce it in a fresh isolated process outside the project, finish notes/navigation/AGENTS/manifests, preserve prior phase bytes and push main. The old app/artifacts stay available as reference.

## Existing preparation helpers

Prepared, guarded scripts live in C:/Users/JACKIV/OneDrive/문서/ChatGPT/SLM 2/quality-recovery-preparation. Read each before running. `completed_training()` chooses 03b_training.json once the continuation is declared, so final-result helpers fail while it is incomplete.

- plot_results.py: real combined curve, with the 16,000 schedule boundary; first-run chapter uses separate preserved initial-learning.png.
- commit_continuation_outcome.py: writes the actual second-stage teaching chapter, then commits the actual continued result (refresh the lesson manifest after authoring if needed).
- write_completed_results.py: requires completed continuation and final metrics; writes measured notes and recovery docs.
- promote_reviewed_model.py: requires truthful all-pairs review and compatible improvement, then copies staged app files and descriptor. Never run publish.py again: its broad copier would prematurely install prepared app changes.
- observe_app.py: actual withdrawn Tk/CPU flow, supplied Save path; no interactive human review.
- source/scripts/package_quality_app.py: copied to active scripts during promotion; run on clean committed source.
- observe_final_bundle.py: fresh isolated existing-runtime relocation/reproduction, not a clean second-machine install.
- close_project.py, tidy_new_prose.py, refresh_recovery_manifest.py: finish navigation and external lesson hashes; inspect archive guards before repeat runs. Preserve earlier phase01–09 chapters.

Canonical new notes: C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes/quality-recovery. Teaching remains concept-and-flow first: missing capability → prerequisite → mechanism/example → return to build → observed limitation. Explain losses with actual populations/masks/denominators/schedules, not just metric tables. Learner mastery and independent human review remain pending.
