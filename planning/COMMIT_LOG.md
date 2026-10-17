# Actual StorySeed history

| Commit | Change |
|---|---|
| `a332e7a` | chore: initialize fresh StorySeed project |
| `f676586` | docs: establish StorySeed learning and commit workflow |
| `b703dda` | docs: record successful initial GitHub publication |
| `5773fd9` | chore: establish the PyTorch environment and inspect gradients |
| `43da6dd` | feat: turn toy stories into inspectable next-token examples |
| `322c2c4` | feat: assemble aligned tensor windows and repeatable batches |
| `84a001a` | feat: implement a trainable bigram score table |
| `d1cb25d` | feat: measure next-token error and inspect cross-entropy gradients |
| `3377bd4` | feat: train the character bigram model from random weights |
| `e60b4db` | feat: sample bounded text continuations from learned scores |
| `5326d0c` | feat: save and reload the trained model in a fresh process |

The current change cannot contain its own hash; use git log to locate it. Add its known hash during the next real change.

## Phase 2 implementation commits

| Stage | Commit | Change |
|---|---|---|
| 2.1 | 25609f8 | Corpus provenance |
| 2.2 | 39ce747 | Whole-story splits |
| 2.3 | 532f7d9 | Explicit BPE |
| 2.4 | bd8c46b | Frozen tokenizer |
| 2.5 | e62fe7e | Boundary-aware batches |
| 2.6 | 1a39605 | Comparable trained baseline |

## Phase 3 implementation commits

| Stage | Commit | Change |
|---|---|---|
| 3.1 | b700b50 | Shared trainable features |
| 3.2 | a0fc359 | Ordered contextual predictor |
| 3.3 setup | 737ae66 | Predeclared training protocol |
| 3.3 outcome | 5ed57b1 | Actual learning and baseline gap |
| 3.4 | effd9d4 | Causal generation and information boundary |

## Phase 4 implementation commits

| Stage | Commit | Change |
|---|---|---|
| 4.1 | 0750d9a | Retrieval-role projections |
| 4.2 | 8b727f4 | Scaled position matches and values |
| 4.3 | 4e0e7ea | Causal/padding restrictions |
| 4.4 | 430d0b6 | Learned window positions |
| 4.5 setup | 3489023 | Integrated model and declared pilot |
| 4.5 outcome | 0a436a4 | Actual learning, information flow and restored inference |
