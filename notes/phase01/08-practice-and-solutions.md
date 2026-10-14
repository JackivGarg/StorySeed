# 1.8 practice: explain the inference bundle

1. Why must the vocabulary be saved in order rather than as an unordered set?
2. Does loading a checkpoint count as training from scratch?
3. Why can our checkpoint generate but not exactly resume Adam training?

## Worked answers

Rows and columns refer to specific token IDs. An unordered vocabulary could assign new IDs, causing the saved scores to refer to different characters. Reconstructing identical mappings preserves the meaning of learned numbers.

No. The checkpoint's weights were trained from random initialization in the earlier run. Loading restores those learned values. Retraining from scratch means initializing a new model and performing updates again; inference does neither.

Adam's moving gradient statistics and update count influence subsequent updates. This inference payload does not restore them or full data/random-state continuation, so a fresh optimizer would behave differently. A complete training checkpoint requires additional state and a defined resume protocol.

Explain the pipeline aloud: text becomes IDs; a current ID selects a learned score row; softmax and configured sampling choose a successor; generation appends it; saved weights and the exact tokenizer let another process repeat the same operation.
