# Phase 3 — learned contextual baseline, actual outcome

Completed on 2026-10-17: all four build-and-teach subphases. The external phase03 notes follow project need -> prerequisite -> concept/example -> return to build -> observation -> next limitation.

## What now works

- A shared 512-by-32 embedding with explicit zero padding behavior.
- An ordered fixed-window ANN: concatenate token features, 64-unit tanh hidden layer, 512 output logits. No attention or Transformer blocks yet.
- Story-aware context gathering over the unchanged Phase 2 arrays; all 359,933 training and 39,711 development target positions remain comparable.
- Fixed-budget random-initialized K=1/8 neural pilots: 300 Adam updates, 256 uniformly sampled targets per update, shared target-index RNG, no development checkpoint selection.
- Local inference-only checkpoints and a common prediction/generation interface.
- A boundary experiment: changing history outside the final eight IDs cannot change K=8 logits; changing an inside-window fixture gave maximum logit change 0.829813.
- Future input changes leave earlier outputs unchanged. Sequence versus repeated-prefix output difference was 2.3842e-6.
- A fresh CLI process reproduces the recorded seeded sample, without retraining or changing the checkpoint hash.

## Measured comparison

| Model | Stored parameters | Final training NLL | Development NLL | Development perplexity |
|---|---:|---:|---:|---:|
| Phase 2 bigram | 262,144 | 3.152857 | 3.287348 | 26.771764 |
| Neural K=1 | 51,776 | 3.847273 | 3.867754 | 47.834826 |
| Neural K=8 | 66,112 | 3.916914 | 3.951838 | 52.030905 |

Both neural models learned from initial development NLL near 6.27. Neither beat the bigram, and K=8 did not beat K=1. These are single-seed source-prefix pilot results with different capacities/optimization budgets; do not claim a context-only causal effect or improved story quality. Generated samples remain incoherent. The original result is retained rather than tuned away.

Recorded neural compute durations after setup were about 4.50/5.91 seconds on CPU, including evaluation/save/reload work. They are not full setup/runtime or GPU throughput benchmarks.

## Evidence and reproduction

01_embeddings.json, 02_context.json, 03_training_k1.json, 03_training_k8.json, 04_limits.json and 05_cli_reload.json contain commands, settings and actual observations. configs/phase03_pilot.json predates execution. Small checkpoint manifests are in data/manifests/phase03; actual weights stay in ignored checkpoints/phase03. Checkpoints do not include exact optimizer resume state.

The separate Desktop StorySeed-Study-Notes/phase03/COVERAGE.md lists all run commands and concepts. To generate without fitting:

```powershell
.venv/Scripts/python scripts/generate_phase03.py --prompt "Once upon a time" --json
```

The final holdout remains unencoded/unscored. Earlier phase notes/reports are preserved. Authored lessons and assistant execution do not establish student mastery. Next project need: content-dependent retrieval across a causal prefix in Phase 4.
