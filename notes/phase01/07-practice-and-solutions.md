# 1.7 practice: generation does not learn new weights

1. Which part determines probabilities, and which chooses among them?
2. Why can top-k=1 repeat forever unless a length budget stops it?
3. Why can equal final characters make two different prompts generate identical suffixes here?

## Worked answers

The trained score row plus temperature/filtering determines probabilities. A categorical random draw chooses among them. The model parameters were learned earlier; no optimizer step runs inside generation.

A bigram has a finite set of current-token states. Deterministically choosing one successor can enter a cycle that never selects `<eos>`. A finite output budget guarantees termination even if ending probabilities or decoding choices are unsuitable.

This architecture ignores earlier prompt positions. With the same last token, weights, settings and starting random-generator state, each step receives the same distribution and random sequence, giving the same selected suffix. Larger prompt strings do not automatically create more context.

For a worked temperature check, `[ln(3),0]` at τ=0.5 becomes `[ln(9),0]`, so probabilities are `[0.9,0.1]`. Scaling logits acts before normalization; directly multiplying already-normalized probabilities is a different operation.
