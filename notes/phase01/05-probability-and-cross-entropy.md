# 1.5: turn a prediction into a differentiable error

The bigram table produces 17 scores at each position. Nothing yet says whether those scores fit the observed answer. We now define that connection. This chapter leads from probability to loss and then to the derivative that will update the table. Read [loss.py](../../src/storyseed/loss.py), [the numerical example](../../lessons/phase01/05_loss.py), [its output](../../reports/phase01/05_loss.md), and [practice](05-practice-and-solutions.md).

## Revise classification using the next-token task

An ANN classifier assigns scores to possible classes. Here the classes are vocabulary token types. The target is one integer class ID, not the next token's embedding or a floating-point score. A language model repeats this classification question at many positions.

For one position, let z_j be the logit for candidate j. Softmax converts relative scores to a distribution:

```text
p_j = exp(z_j) / sum_k exp(z_k)
```

The sum ranges over all vocabulary candidates k. Every probability is nonnegative and the probabilities sum to one. Softmax along the vocabulary axis is essential: normalizing across batch examples or time positions would answer a different question.

## A calculation with three choices

Use logits `[ln(2),0,0]`, where ln is the natural logarithm. Exponentials become `[2,1,1]`, their sum is 4, and probabilities are `[0.5,0.25,0.25]`. Choose target ID 0. Its negative log-probability is:

```text
L = -ln(p_target) = -ln(0.5) = ln(2) ≈ 0.693147
```

If the target score becomes ln(4), probabilities are `[4/6,1/6,1/6]`. The target probability increases to 2/3, so loss becomes `-ln(2/3) ≈ 0.405465`. Our script verifies both numbers. This does not update a model: we are manually comparing two possible score vectors.

If all three scores are equal, each probability is 1/3 and loss is ln(3). For our 17-choice vocabulary, equal scores give loss ln(17), approximately 2.833213. That is a useful reference for initial near-uniform predictions, not a universal accuracy baseline.

The logarithm penalizes assigning very low probability to the observed target. A probability of 0.9 has a much smaller negative log than 0.01. Lower loss means better token prediction under this objective. It does not prove that sampled text forms coherent stories.

## Derive the score gradient

Because `p_y = exp(z_y)/sum exp(z)`, substitute it into the loss:

```text
L = -z_y + ln(sum_k exp(z_k))
dL/dz_j = p_j - indicator(j == y)
```

The indicator is 1 for the observed target candidate and 0 otherwise. For our probabilities and target 0, the derivative is `[-0.5,0.25,0.25]`. Gradient descent subtracts this vector: it raises the target's relative score and lowers the competing scores. The changes are coupled through normalization.

This derivative explains the role of a label without guessing how the optimizer works. When many examples select the same bigram row, their contributions add to that row's gradient. Conflicting observed successors therefore produce a distributional learning problem rather than requiring one deterministic answer.

Adding the same constant to every logit leaves softmax unchanged. The model cares about relative scores. This is why a large positive individual number cannot be interpreted as high confidence without considering the other logits.

## Trace our tensor implementation

Our model returns logits `[B,T,V]`; targets are `[B,T]`. Cross-entropy expects a list of classification examples with V scores each. We reshape to `[B*T,V]` and target shape `[B*T]`, preserving the same row order. For B = 2, T = 3, V = 17, six positions each contribute 17 scores and one target ID.

`next_token_loss` validates the matching shapes and integer target type, then uses the built-in cross-entropy operation on **logits**. Its mean reduction averages over the contributing positions. Our Phase 1 examples are unpadded, so every position contributes. Explicit ignored-label and masking policies will be studied when needed for padded data.

Do not apply softmax before calling this loss. It already performs the appropriate score-to-log-probability calculation. Giving probabilities as though they were logits changes the computation; the example records a different incorrect value. Internally using a stable log-softmax calculation also avoids unnecessary direct exponentiation of extremely large scores.

The script separately computes each position's loss and averages them, verifying agreement with flattening. This checks that the tensor reshaping preserved the intended questions. Equal shapes alone could not prove target order was correct.

## What happens next

We have a prediction function and a differentiable error. Backpropagation now has a route from that error to the parameter table. The next chapter uses actual optimization to change W and watches both training and development behavior. Before proceeding, explain why the target is an ID while the model output is a full score vector.

API reference: [PyTorch cross-entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html). The worked three-choice calculation and score-gradient derivation above are teaching examples for our implementation.
