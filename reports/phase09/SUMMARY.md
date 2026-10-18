# Phase 9 — local application, release and independent reconstruction

Five focused increments extend the existing end-to-end plan. There is no new training or final-set scoring in this phase.

| Increment | Delivered capability | Evidence / learning |
|---|---|---|
| 9.1 | Fixed-main CPU generation service and CLI | 01_generation.json, 01_cli.json; same generated IDs/text as Phase 7 |
| 9.2 | Tk desktop input, controls, output, copy and JSON save | 02_desktop.json; actual withdrawn Tk loop drove model load/submit/display/copy; save dialog and independent usability review not exercised |
| 9.3 | Source/weight/tokenizer bundle with integrity manifest | 03_package-*.json and 03_relocated-*.json; isolated fresh process outside repo used included runtime and reproduced IDs |
| 9.4 | Model card, university report and demo guide | docs/MODEL_CARD.md, UNIVERSITY_REPORT.md, DEMO_GUIDE.md; evidence and pending human ratings stated explicitly |
| 9.5 | Connected whole-system reconstruction workbook | docs/RECONSTRUCTION_WORKBOOK.md and separate Desktop phase09 notes; student demonstration pending |

Start locally: `.venv/Scripts/python -m storyseed.app --gui`, or use `scripts/Start-StorySeed.ps1`. CLI: `.venv/Scripts/python -m storyseed.app --json`. Setup and bundle details are in docs/LOCAL_APP.md. Check FINAL_RELEASE.json for the final complete local bundle path/digest after closure.

## Model interpretation

The app retains the predeclared 6,605,312-parameter main128 export with its exact Phase 2 tokenizer. It uses CPU FP32, one thread, default temperature 0.8, top-k 40, seed 17 and up to 48 new IDs. The same existing sampler implements command-line and desktop requests. Prompt cropping is reported; the model directly sees only the newest 128 input IDs. Generation does not update weights.

Main full-development NLL was 6.295087 → 2.805321 during Phase 7. Its fixed-final NLL was 2.777847; the context64 sibling scored 2.609649 on the same 44,282 target events. That one-seed finding remains intact. Packaging and a polished interface do not improve those learned results, and no new loss curve is invented. Generated grammar/continuity errors remain visible.

## Boundaries

The bundle includes trained tensors, tokenizer, runtime source and documentation, not Python, third-party dependency binaries, raw stories or full optimizer checkpoints. Dependencies must already be installed or installed separately; generation is local after setup. Fresh-process relocation was observed on the existing laptop/runtime, not a clean second-machine installation. The manifest is unsigned, so its hashes are integrity/compatibility checks rather than publisher authentication.

Independent human quality ratings, visual/usability review, other-machine installation, independent student reconstruction and publication acceptance remain pending/unclaimed. No automated test suite was added or run for this phase. Earlier phase01–08 chapters and historical results are preserved. The original final evaluation population is now used evidence; additional tuning needs a new declared evaluation design.
