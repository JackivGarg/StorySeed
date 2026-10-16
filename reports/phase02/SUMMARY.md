# Phase 2 — measured pipeline outcome

- Captured 1,048,576 source bytes at a pinned revision: 1,129 complete stories, incomplete tail discarded.
- Split whole normalized groups: 904 training, 105 development, 120 reserved final holdout; no overlap under the declared grouping rule.
- Explicit eight-merge BPE fixture reconstructs its inputs and Unicode probes.
- Main tokenizer: 512 entries, tokenizers 0.22.1, training-only fit; exact round-trip/reload checks for 1,009 train/development stories plus three additional probes.
- Main tokenizer hash: 3540aba1c244ed063b9fb317f0d62280a63d00725cac980e16c3b583cdf549cb.
- Train/development true target positions: 359,933 / 39,711, including one EOS target per story.
- T=64 chunks: 6,050 / 673. Right padding is excluded from loss; document boundaries are retained.
- The 262,144-parameter bigram trained from random scores using 200 CPU Adam updates on the exact count-grouped objective.
- Train NLL: 6.238044 → 3.152857. Development NLL: 6.238020 → 3.287348. Final development perplexity: 26.771764.
- Development NLL at step 100 was 3.283216, slightly better than the fixed-budget final result. This is reported, not hidden.
- Smoothed unigram/bigram development NLL: 5.221827 / 3.287084, alpha=0.1.
- Saved weights, tokenizer identity and seeded output were restored; a fresh CLI process matched the recorded sample.
- The final holdout remains unencoded/unscored. No final-transformer or student-mastery claim is made.

## Scope and reproduction

These are teaching-pilot results from a source prefix, not representative full-corpus scores. Phase 1 uses different data/token units and its numeric losses are not directly comparable.

Detailed actual observations are in 01_corpus through 06_baseline JSON/Markdown reports. CLI reproduction evidence is 06_cli_generation.json and 06_cli_reload.json. The source, split, tokenizer, encoded and baseline manifests live under data/manifests/phase02.

The trained checkpoint is local and ignored by Git. The small tokenizer JSON is versioned without Git text conversion to preserve its exact hash. A new tokenizer/corpus contract requires compatible arrays and baseline refitting.

External teaching is in StorySeed-Study-Notes/phase02. Next modelling need: learned representations and richer usable context in Phase 3.
