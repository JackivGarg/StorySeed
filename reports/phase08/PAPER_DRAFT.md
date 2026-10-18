# StorySeed: A Resource-Constrained Study of Context Policies in a Scratch-Trained Small Language Model

Status: university research draft based on actual local execution. Independent human evaluation is pending. No venue submission or acceptance is claimed.

## Abstract

We implement a compact causal decoder explicitly in PyTorch and train it from random initialization for simple English story continuation. We compare 64 versus 128 supplied input slots while holding maximum position-table size, parameter count, initial weights, training row draws and successful target/update budgets fixed. The experiment uses one paired training seed and a bounded TinyStories source prefix. We report full-vocabulary next-token NLL, perplexity, lexical repetition, measured resources and a paired review packet. The study demonstrates a reproducible educational experiment, rather than a new language-model architecture.

## 1. Motivation and related work

Transformers introduced attention-based sequence processing [1]. TinyStories studies language learning in much smaller models using constrained synthetic stories [2]. StorySeed follows established techniques and asks a local resource/conditioning question: how do two supplied-context policies behave under an equal target budget? It does not claim to reproduce the original paper's training scale or results.

## 2. Data and implementation

We capture the first 8 MiB of TinyStories-train.txt at revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64. Training uses 7,585 normalized-unique groups assigned train by the unchanged seed-17 content-hash rule. The original 105 development and 120 final stories remain separate. The unchanged 512-entry byte-level BPE tokenizer was fit using only the earlier training subset. This small vocabulary increases sequence fragmentation and is a limitation, not a claim of an optimal tokenizer. Exact/normalized overlap checks do not rule out semantic near duplicates. Source-prefix sampling is not representative of the full dataset.

The model has 6,605,312 parameters: width 256, eight heads, eight pre-LayerNorm decoder blocks, GELU feed-forward width 1,024, learned 128-position embeddings and untied vocabulary weights. Parameters begin randomly. PyTorch supplies differentiation/device infrastructure; no pretrained weights are used.

## 3. Predeclared method

Each condition completes 2,000 AdamW updates with four-row microbatches accumulated four times, scored width 64, warm-up 100, maximum LR 0.0003 and minimum LR 0.00003. Both use the same seed 17, data seed 2026, dropout 0.1 and successful target budget. CUDA BF16 autocast uses FP32 master weights and loss. The last checkpoint is selected in advance; development is measured every 250 updates. A final set is encoded/scored only after both exports are frozen.

With 64 slots, a row supplies up to its 64 scored inputs. With 128 slots, up to 64 additional earlier story inputs are prepended. This policy does not provide a separately maximal prefix at every target. Initial weights, rows and target counts match; compute does not. Window-relative position usage and dropout realizations vary with input length. This is an intervention on the implemented context policy, not a universal architectural claim.

## 4. Results

| Model | Final NLL | Perplexity | Target positions |
|---|---:|---:|---:|
| main128 | 2.777847 | 16.084 | 44282 |
| context64 | 2.609649 | 13.594 | 44282 |
| expanded_count_bigram_alpha_0.1 | 3.201461 | 24.568 | 44282 |
| historical_phase02_learned_bigram | 3.265621 | 26.196 | 44282 |

Main full-development NLL changes from 6.295087 to 2.805321. This is a fixed-population predictive measurement, distinct from noisy pre-update training batches. The paired final NLL difference is 0.168199; descriptive paired story-bootstrap interval [0.155971, 0.180609]. This interval conditions on one trained pair and does not capture training-seed variability.

Raw synchronized update-wall totals are preserved, but the short-context total includes the recorded isolated stall and is not a clean speed comparator. A separate resource-only observation uses fixed rows, three trials per policy, three warm-up and ten timed updates per trial. The following are descriptive short-workload measurements; GPU clocks were not fixed, and no intrinsic speed advantage is claimed.

| Condition | Scientific-run target presentations | Separate matched median update seconds | Median trial scored targets/second | Matched-observation peak allocated bytes |
|---|---:|---:|---:|---:|
| main128 | 1,884,373 | 0.767862 | 1183.26 | 213,870,592 |
| context64 | 1,884,373 | 0.799871 | 1145.52 | 167,141,888 |

The large-stall validity check passed, but that criterion does not establish clock control or statistical certainty about small timing differences. Longer inputs have higher analytical work/storage demands; measured wall duration also depends on hardware utilization and power state. See matched_resources.json and the retained execution incident.


Twelve sampled continuations per condition use six predeclared prompts and seeds 17/29, temperature 0.8, top-k 40 and maximum 128 new IDs. Individual samples, EOS/length and lexical metrics are preserved. Grammar/continuity/consistency judgments require review; no human scores are supplied here.

## 5. Limitations

The dataset is a small convenience prefix, the tokenizer was learned on a smaller training subset, and there is one paired training seed. Hyperparameter choices are not claimed optimal. The historical baseline differs in data exposure/budget; the count baseline has a different inductive bias/fit procedure. Word repetition is not narrative coherence. The fixed held-out evaluation must not be reused for further tuning. No independent human ratings, agreement analysis, broad benchmark superiority, new algorithm or publication acceptance is claimed.

## 6. Conclusion

The project produces traceable scratch-trained weights and a controlled local comparison with honest limits. The evidence supports the measured conditional findings above and provides a basis for additional seeds, independent story review and larger-data work. It does not establish general superiority from one run.

## References

[1] Vaswani et al. (2017), Attention Is All You Need. https://arxiv.org/abs/1706.03762

[2] Eldan and Li (2023), TinyStories: How Small Can Language Models Be and Still Speak Coherent English? https://arxiv.org/abs/2305.07759

[3] Pinned TinyStories dataset card (source/declared terms): https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md

[4] Perplexity and finite-context evaluation: https://huggingface.co/docs/transformers/perplexity

## Reproduction and artifacts

The protocol and its byte hashes are in planning/PHASE07_08_PROTOCOL.json and PHASE07_08_FREEZE.json. Source/configuration, provenance, split/tokenizer identities, condition reports, full curves, final metrics, generated samples and review forms are versioned; raw text and tensor checkpoints remain local and ignored. The main export SHA-256 is `f0f7241af4770372738c3c9b057e205cc60e66cc7fa1d21397a6281f383c0eed`. The finite foreground command is `.venv/Scripts/python scripts/run_phase07_08.py`; re-executing it restores valid unfinished states or reads completed compatible records, rather than silently changing the protocol.

## Operational timing incident and recovery

The context64 process stopped at a valid checkpoint after its runtime guard fired. One recorded update contained approximately 17,768 seconds of synchronized wall delay, while the median preceding update was approximately 0.182 seconds. The cause of that isolated system/device stall was not established. The execution allowance was explicitly amended before final-set access; data, optimizer settings, RNG checkpoint, target/update budgets and last-artifact selection stayed fixed. The unfinished condition resumed from update 669.

Raw timing retains the stall; gross condition durations must not be interpreted as a clean context-speed comparison. A separate matched resource-only observation uses scratch copies, fixed rows, three trials per context, three warm-up and ten timed updates per trial. All samples are retained, and an update above the predeclared five-second threshold marks that observation contaminated. See matched_resources.json for actual validity and measurements. This resource observation does not change the frozen trained artifacts or their quality scores.

## Identified qualitative review

The assistant reviewed all 24 retained continuations and recorded pair-specific observations in ASSISTANT_SAMPLE_REVIEW.md. Both conditions show familiar story openings but malformed words, prompt drift and inconsistent references. Main128's zero duplicate-trigram fraction did not imply coherence; generic opening templates recur across prompts. Context64's four EOS stops did not necessarily resolve the prompt's plot. These are identified assistant observations, not independent human ratings. The human rubric remains pending.
