# Observed fixed-protocol evaluation

The final population was encoded only after both predeclared 2,000-update model exports were fixed. No additional training or selection used these results.

| Model | Final NLL | Perplexity | Target positions |
|---|---:|---:|---:|
| main128 | 2.777847 | 16.084 | 44282 |
| context64 | 2.609649 | 13.594 | 44282 |
| expanded_count_bigram_alpha_0.1 | 3.201461 | 24.568 | 44282 |
| historical_phase02_learned_bigram | 3.265621 | 26.196 | 44282 |

![Actual frozen final loss](C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes/phase08/assets/final-loss.png)

The paired context difference is main128 minus context64: 0.168199 nats/target. Its descriptive story-bootstrap 95% percentile interval is [0.155971, 0.180609]. Negative values favor main128 under the stated supplied-context policy. The resampling holds trained weights fixed and varies stories; it does not estimate uncertainty across training seeds. There is only one paired training seed.

Equal parameter count, initial model state, sampled training rows, successful updates, true-target presentations and final target counts were checked in the recorded execution. Compute is not matched: longer inputs require more processing. Window-relative positions and dropout draws change with supplied input length, so this is a comparison of context policies in our implemented system.

The expanded count bigram sees exactly the expanded training population and uses fixed additive smoothing alpha 0.1. The historical Phase 2 learned bigram used less training data and a different optimization budget. Their scores are useful references, not an architecture-only capacity-matched experiment.

The paired continuation packet contains 24 generated samples from six frozen prompts and two sampling seeds per condition. Word-trigram repetition and lexical diversity measure observable surface properties. They do not measure narrative coherence, and their interpretation depends on output length and decoding settings. Human rating fields remain blank and pending. A future assistant review must identify itself rather than being described as an independent human study.

See the actual `reports/phase08/metrics.json`, `samples.json`, `PAIRED_SAMPLE_REVIEW.md` and `PAPER_DRAFT.md` in the code repository. The final set is now used under this frozen protocol; future tuning needs a new untouched evaluation population. Learner understanding remains pending.

## Operational timing incident and recovery

The context64 process stopped at a valid checkpoint after its runtime guard fired. One recorded update contained approximately 17,768 seconds of synchronized wall delay, while the median preceding update was approximately 0.182 seconds. The cause of that isolated system/device stall was not established. The execution allowance was explicitly amended before final-set access; data, optimizer settings, RNG checkpoint, target/update budgets and last-artifact selection stayed fixed. The unfinished condition resumed from update 669.

Raw timing retains the stall; gross condition durations must not be interpreted as a clean context-speed comparison. A separate matched resource-only observation uses scratch copies, fixed rows, three trials per context, three warm-up and ten timed updates per trial. All samples are retained, and an update above the predeclared five-second threshold marks that observation contaminated. See matched_resources.json for actual validity and measurements. This resource observation does not change the frozen trained artifacts or their quality scores.
