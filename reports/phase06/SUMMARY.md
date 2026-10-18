# Phase 6 outcome — 2026-10-18

Six natural implementation-and-teaching increments build a dependable training process around the complete decoder. Sources, configs, individual reports and actual commits are traceable through planning/LESSON_MAP.md and planning/COMMIT_LOG.md.

## Actual bounded execution

- The story-local warm-up/scored-block policy covers each development target once; the initial full score is 6.246756208669128 nats/target over 39,711 events. Context policy differs from earlier per-target-window evaluation and is disclosed.
- Correct whole-token gradient accumulation agrees with whole-batch gradients to 2.98e-8 maximum absolute difference in the controlled fixture. Deliberately averaging unequal microbatch means differs by 0.649415.
- FP32/FP16/BF16 updates actually execute on the RTX 3050 Laptop GPU; master parameters/loss remain FP32 where specified. The controlled FP16 infinite-gradient attempt skips the optimizer, keeps weights/counter unchanged and reduces scale 1024 → 512.
- Eight uninterrupted CPU updates versus three plus five in a fresh CLI process match model, optimizer, Torch/data RNG and remaining row/loss/LR history bitwise in that recorded environment. Weights-only continuation differs by maximum parameter delta 0.0259852. This is not a cross-runtime equivalence guarantee.
- The actual public CPU CLI performs ten successful updates (2,495 sampled target presentations), periodic checkpoints, full evaluation and inference export. Development NLL becomes 5.843922512457613; perplexity 345.1304675851018. The exported sample remains incoherent. Fresh-process seeded inference matches.
- The separate width-256/eight-layer/eight-head/context-128 CUDA BF16 candidate has 6,605,312 parameters. Latest timed three updates: 3,008 scored targets / 1.458355 seconds = 2,062.60 scored target tokens/sec. Peak allocated PyTorch memory: 213,870,592 bytes. The initial independent timing is preserved too.
- A 2,000-update compute projection is about 972 seconds from this short measurement, excluding evaluation/checkpoint overhead, thermal variation and changed data choices. Main training has not been launched; corpus and main protocol remain to freeze in Phase 7.

## Artifacts and study flow

Local tensors stay ignored. Small identities: data/manifests/phase06/resume-checkpoint.json and pilot-artifacts.json. The separate Desktop phase06 folder contains six concept chapters, six reasoning companions, a guide, reconstruction, coverage, actual commit map and detailed loss interpretation. Charts draw only the recorded ten sampled losses and the two full-development measurements; the small report 08_learning_observations.json preserves their source identity.

The explanation separates sampled pre-update NLL, full-population post-update development loss, regularization, numerical precision and continuation state. Metrics from different protocols/budgets are not a clean architecture ranking; no quality win is claimed.

Earlier Phase 1–4 chapters remain unchanged. Final holdout remains unencoded/unscored. No automated test suite was added or run. Authored teaching and assistant execution are complete; learner understanding remains pending. Next: Phase 7 main protocol and training, followed by Phase 8 controlled evaluation and Phase 9 packaging/independent reconstruction.
