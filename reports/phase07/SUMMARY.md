# Observed main-training outcome

Generated from completed records; no values are projected. The main128 model began from random weights, with 6,605,312 parameters, and completed 2,000 successful updates. Training uses 7,585 stories from the pinned 8 MiB source prefix and the unchanged 512-entry tokenizer.

Full development NLL moved from 6.295087 to 2.805321 on the same 39,711 targets. The training process presented 1,884,373 true targets, equivalent to 0.669 times the available training target population. Sampling is with replacement, so that ratio is not proof that every position was visited.

![Actual main learning records](C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes/phase07/assets/main128-learning.png)

Read the curve by its measurement contract: the fine training trace uses changing sampled rows with dropout before each update. The development markers use a fixed population, evaluation mode and completed-update weights. An upward training point need not mean the model forgot; a persistent development rise needs investigation. The individual recorded markers, not a smooth invented trajectory, support interpretation.

Learning changes the parameters through accumulated token gradients and AdamW. The architecture, tokenizer and underlying labels stay fixed during each run. No component ablation isolates how much of this improvement came from a particular head, sublayer or schedule. Inspect actual samples in the curve record to connect predictive improvement to visible generation failures.

The exported artifact is `C:\Users\JACKIV\OneDrive\Desktop\StorySeed\checkpoints\phase07\main128-inference.pt`, SHA-256 `f0f7241af4770372738c3c9b057e205cc60e66cc7fa1d21397a6281f383c0eed`. The public generator reproduces its seeded output in a fresh process; it does not train. Full resume checkpoints contain optimizer/random/data state, while the inference export does not.

The companion context64 run is a separate random-initialized training condition, not a shortened view of the already-trained main model. Both runs are used by the frozen Phase 8 protocol. Learner understanding remains pending.

## Operational timing incident and recovery

The context64 process stopped at a valid checkpoint after its runtime guard fired. One recorded update contained approximately 17,768 seconds of synchronized wall delay, while the median preceding update was approximately 0.182 seconds. The cause of that isolated system/device stall was not established. The execution allowance was explicitly amended before final-set access; data, optimizer settings, RNG checkpoint, target/update budgets and last-artifact selection stayed fixed. The unfinished condition resumed from update 669.

Raw timing retains the stall; gross condition durations must not be interpreted as a clean context-speed comparison. A separate matched resource-only observation uses scratch copies, fixed rows, three trials per context, three warm-up and ten timed updates per trial. All samples are retained, and an update above the predeclared five-second threshold marks that observation contaminated. See matched_resources.json for actual validity and measurements. This resource observation does not change the frozen trained artifacts or their quality scores.
