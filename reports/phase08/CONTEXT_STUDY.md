# Interpret the actual context-policy result

Both artifacts were fixed at update 2,000 before the original 120 final stories were encoded. The evaluation changes what we know about the models; it makes no optimizer update. The final population is now used evidence and must not be recycled as untouched data for further tuning.

| Fixed artifact | Final NLL | Perplexity |
|---|---:|---:|
| main128 | 2.777847 | 16.084 |
| context64 | 2.609649 | 13.594 |
| expanded_count_bigram_alpha_0.1 | 3.201461 | 24.568 |
| historical_phase02_learned_bigram | 3.265621 | 26.196 |

These are newly computed scores on identical final next-ID events, including EOS and excluding PAD/warm-up duplicates. They are not copied development numbers from earlier phases. Their common tokenizer makes their units compatible, but the historical bigram still has different training data/budget. The expanded count bigram sees all expanded training transitions through its counting procedure, unlike the decoder's sampled finite exposure.

The main model's last development NLL is 2.805321, while its final-set NLL is 2.777847. Those use the same frozen weights but different story populations. Their difference is not another learning step: final evaluation performs no backward/optimizer update. This is why every loss change needs its measurement context before we explain what caused it.

## What the controlled pair says

The signed main128-minus-context64 NLL difference is +0.168199: context64 has lower observed NLL. The paired story-bootstrap percentile interval is [+0.155971, +0.180609], with 2,000 replicates. Interpret its sign conditional on these two fixed trained models. It does not estimate variability across training initializations, because only one paired training seed was run.

Equal parameter count, initial parameter hash, actual row-sequence hash, successful update count, target presentations and final target counts were checked. The longer policy supplies more earlier story inputs, but also changes window-relative position usage and shape-dependent dropout draws. We interpret the intervention as the implemented supplied-context policy, not an isolated universal memory law.

## What it cost

| Condition | Scientific-run target presentations | Separate matched median update seconds | Median trial scored targets/second | Matched-observation peak allocated bytes |
|---|---:|---:|---:|---:|
| main128 | 1,884,373 | 0.767862 | 1183.26 | 213,870,592 |
| context64 | 1,884,373 | 0.799871 | 1145.52 | 167,141,888 |

The original context64 run contains a recorded 17,768-second synchronized wall-delay outlier. Its gross update durations are retained in the raw history and result, but are contaminated for a clean context-speed conclusion. We amended only the execution allowance before final access, restored the valid update-669 checkpoint and completed the unchanged scientific target/update budget.

The table therefore uses the separately predeclared matched resource observation: three trials per condition, three warm-up and ten timed updates per trial, fixed row IDs and resource-only scratch copies. Its validity flag is `True`; a false flag means no clean speed conclusion should be drawn even from that observation. The median shown is a summary of the retained trial measurements, not a deletion of the raw long-run incident.

A true flag checks only the stated large-stall criterion; it does not prove clocks were held constant or that a small timing difference is a robust speed advantage. In these observations the two update medians are close despite different analytical work counts. The device record shows a low-power P8 state before measurement. We report the numbers descriptively, with no intrinsic algorithmic speed ranking inferred from them.

Longer inputs require more state processing and larger attention matrices. Actual runtime does not have to follow the attention-matrix ratio alone because projections, feed-forward work, kernel overhead and hardware utilization also matter. Compute is measured rather than held equal. Peak allocation is active PyTorch memory in the stated workload, not total device memory or a guarantee that every larger batch fits.

## Connect the score to the stories

The packet retains all 24 frozen prompt/seed continuations. Repetition, lexical diversity, EOS and output length are recorded as surface properties. High diversity is not coherence, and short outputs can distort those ratios. Read the identified assistant qualitative review alongside the actual text. Independent human scores remain pending in the blank rating form; no human means or agreement statistics were invented.

The main model has zero mean duplicate-trigram fraction within outputs, yet its samples contain malformed words and unclear actions. Several seed-29 prompts reuse a generic outdoor/park opening. Exact within-output repetition therefore misses prompt drift and cross-output template reuse. Context64 emits EOS in four of twelve cases; the reviewed examples show that an end marker does not establish a resolved plot.

The university draft reports the local result and limitations. It does not claim a new architecture, broad superiority or publication acceptance. Future training-seed repeats and identified human review would strengthen a later study, but should use appropriate fresh evaluation discipline.

### What I should now understand

Explain the observed signed difference with its actual controls, the conditional interval and resource tradeoff; distinguish baseline exposure; connect likelihood to visible output limits; and say why this result cannot support a universal model-quality claim.
