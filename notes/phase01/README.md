# Phase 1 study guide: follow eight real build commits

Phase 1's implementation and notes are complete. Your understanding remains something we will develop and check together. This guide helps you read the code history as a connected book instead of trying to understand the final repository all at once.

## Read in the order the system grew

| Subphase and commit | Read first | Then inspect |
|---|---|---|
| 1.1 · 5773fd9 | [Environment, tensors and gradients](01-environment-tensors-and-gradients.md) | lessons/phase01/01_environment.py |
| 1.2 · 43da6dd | [Text, tokens and targets](02-text-tokens-and-targets.md) | src/storyseed/tokenizer.py, toy_data.py |
| 1.3 · 322c2c4 | [Windows, batches and shapes](03-windows-batches-and-shapes.md) | src/storyseed/data.py |
| 1.4 · 84a001a | [Bigram scores](04-bigram-scores.md) | src/storyseed/bigram.py |
| 1.5 · d1cb25d | [Probability and cross-entropy](05-probability-and-cross-entropy.md) | src/storyseed/loss.py |
| 1.6 · 3377bd4 | [Training and updates](06-training-and-updates.md) | src/storyseed/training.py |
| 1.7 · e60b4db | [Generation and sampling](07-generation-and-sampling.md) | src/storyseed/generation.py |
| 1.8 · 5326d0c | [Saving and loading](08-saving-and-loading.md) | src/storyseed/checkpoint.py, scripts/generate.py |

For each row, read the chapter, try its companion practice, inspect the commit's diff, and compare the recorded lesson output. The full [chapter index](../README.md) links all eight practice/solution files and run reports.

Open commits on GitHub or use `git show <commit>` locally. Read the chapter at that revision if you want the exact historical state. You do not need to overwrite current files to inspect earlier code.

## What we have learned how to construct

Text is encoded into categorical IDs. Each observed next token becomes a label. The table has one trainable row per current token; each row produces scores over successor choices. Cross-entropy connects those scores to labels. Backpropagation computes gradients, and an optimizer changes the scores. Generation chooses a token from the learned distribution, appends it and repeats. Saving the weights with token ordering lets another process reproduce the same operation.

This is the full language-learning loop in a deliberately small model. It is not the final transformer. Its limited context gives us a reason to study representations and attention rather than introducing them as unexplained buzzwords.

## Read the actual results honestly

The main toy run has 17 token types, 289 parameters, 240 training target positions, 67 development positions and 400 updates. Losses decrease, and the learned distribution approaches the frequency-based optimum for a bigram. A sample such as `mia sat m t.` exposes its remaining weakness. The readable greedy sample `mia sat.` is useful to inspect but cannot replace a systematic evaluation.

The model checkpoint is local and ignored in Git. The scripts let you recreate it. Eight tests verify important contracts, and the checkpoint lesson reproduces inference in a separate process. Complete training resumption, a real story corpus and the transformer architecture remain later work.

## Three questions to connect the whole phase

1. Why do input IDs need a score table before we can predict anything?
2. How does a target affect a selected table row through loss and gradients?
3. Why can the trained table lower its loss yet fail to maintain a story plot?

Work through these in your own words after the chapters. If a step is unclear, tell me the chapter and calculation; we will add a focused explanation before moving to a dependent concept.

## Symbols and terms worth keeping straight

- **V:** number of token types, including the story-ending marker.
- **B, T:** batch examples and positions per example.
- **W:** learned score table, shape `[V,V]`.
- **Logits:** unnormalized scores, not token IDs or probabilities.
- **Loss:** prediction error under a specified objective, not a measure of every aspect of story quality.
- **Gradient:** derivative used to guide an update; backward does not itself apply the update.
- **Checkpoint:** saved values and metadata, distinct from the source code that computes with them.

The next phase changes the data pipeline while preserving these meanings. Review outstanding questions first; no independent mastery is presumed from the completed implementation.
