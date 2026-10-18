# Phase 5 outcome — 2026-10-18

Six focused implementation-and-teaching increments build explicit multi-head causal attention, a GELU feed-forward sublayer, pre-LayerNorm residual blocks, dropout/modes, the complete stack and a bounded tiny-batch learning diagnostic. Natural commits and external note references are in planning/COMMIT_LOG.md and planning/LESSON_MAP.md.

## Actual observations

- Default context-64, width-32, four-head, two-block decoder: 60,032 parameters. It uses learned window-relative positions, pre-LayerNorm, untied vocabulary weights and random initialization.
- Separate context-32 diagnostic: 59,008 parameters; dropout zero; constant-lr Adam 0.01; 200 updates on the same 124 target events (24,800 presentations).
- Same-batch NLL: 6.242737770080566 → 0.0017903411062434316. This is deliberate memorization, not development evaluation or coherent story quality.
- The exercised future-input earlier-logit difference and earlier-objective future-state gradient are both zero. Saved inference weights restore logits and a fresh CLI reproduces the seeded sample.
- Local inference artifact: checkpoints/phase05/tiny-decoder.pt. Its identity is in data/manifests/phase05/tiny-decoder.json; tensors are ignored, not uploaded.

## Teaching and limits

The separate Desktop phase05 folder contains six concept chapters, six reasoning companions, a guide, reconstruction, coverage, actual commit map and detailed loss interpretation with an actual-observation chart. The explanation follows capability need → prerequisite → return to build → observation → limitation. The plot distinguishes pre-update logs from completed updates.

Earlier phase chapters are unchanged. The final holdout remains unencoded/unscored. No automated test suite was added or run. Learner understanding is pending. Phase 6 supplies the dependable training process; main model training is Phase 7.
