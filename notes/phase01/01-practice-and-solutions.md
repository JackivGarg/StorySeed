# 1.1 practice: prediction, loss and a gradient

Try these before reading the worked answers.

1. In our calculation, which quantities are fixed and which is trainable?
2. At `w = 1`, compute the prediction, squared loss, derivative and update with learning rate 0.1.
3. Why does a working CUDA calculation not prove that the main transformer fits in memory?

## Worked answers

The input 3 and target 5 are fixed for this example. The scalar weight is trainable. The prediction and loss are computed values that depend on it.

At `w = 1`, `p = 3`, residual `p-y = -2`, and `L = 4`. The derivative is `2 × (-2) × 3 = -12`. Subtracting it gives `w_new = 1 - 0.1(-12) = 2.2`. The new prediction is 6.6 and loss is 2.56. The update moves in a direction that lowers the loss for this step, but crosses the optimum. This explains why subtraction of a negative derivative increases a weight.

The CUDA example allocates only a few scalars. A transformer has parameters, activations, gradients and optimizer state whose memory depends on architecture, sequence length and batch size. Those must be measured later.

Check yourself: explain why our derivative is 6 at `w=2` without reading the formula. An assistant-provided answer is a learning aid; your own reasoning provides the evidence of understanding.
