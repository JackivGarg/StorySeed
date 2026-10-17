# Phase 4 — causal attention, actual outcome

Five build-and-teach subphases completed on 2026-10-17. Their separate external phase04 chapters follow missing capability -> prerequisite -> concept/example -> return to justified implementation -> observation -> next limitation.

## What works

- Explicit shared Q/K/V projections, scaled pair scores, position softmax and weighted values.
- A diagonal-inclusive causal information rule, source/query padding masks and safe zero output for empty padded query rows.
- Learned token-plus-position states using 0-based real offsets within each supplied window; truncation recenters positions.
- One head, a 64-ID maximum context and a vocabulary head: 28,672 stored parameters. No residual, LayerNorm, feed-forward sublayer, dropout or full decoder stack yet.
- A random-initialized 300-update CPU pilot, with the same frozen tokenizer/splits/target populations and a predeclared final checkpoint.
- Full-forward/final-query agreement within tolerance, zero measured future influence and zero future-state gradient for the selected earlier-position objective.
- An eligible change outside the old ANN's 8-ID window changes the recorded attention logits; changes outside this model's 64-ID window do not.
- Local checkpoint restoration and fresh-process seeded generation without retraining or changing weights.

## Measured results

Training NLL: 6.246869 -> 4.651900. Development NLL: 6.246256 -> 4.653214. Compute after setup: 19.58 seconds, including evaluation/save/reload work, not environment creation or GPU benchmarking.

| Model | Development NLL | Development perplexity | Target positions |
|---|---:|---:|---:|
| phase02_bigram | 3.287348 | 26.771764 | 39711 |
| phase03_k1 | 3.867754 | 47.834826 | 39711 |
| phase03_k8 | 3.951838 | 52.030905 | 39711 |
| phase04_attention | 4.653214 | 104.921623 | 39711 |

This minimal attention pilot learned but did not beat any earlier trained baseline. Samples are incoherent. Capacity, architecture, exposure and compute differ; the ranking is not a context-only controlled effect and does not establish that attention is intrinsically better or worse. The observed result is retained.

## Reproduction and scope

The seven JSON/Markdown report pairs record numerical fixtures, commands, settings and actual observations. configs/phase04_pilot.json predates execution. data/manifests/phase04/attention.json identifies local ignored checkpoints/phase04/attention.pt. The artifact is inference-only and does not contain exact training-resume state.

For inference only:

```powershell
.venv/Scripts/python scripts/generate_phase04.py --prompt "Once upon a time" --json
```

External phase04/COVERAGE.md lists the complete commands and teaching locations. Phase 1–3 chapters/historical reports are unchanged. The final holdout remains unencoded/unscored and learner understanding remains pending. Next: Phase 5 complementary heads and the complete decoder block, followed later by dependable main training.
