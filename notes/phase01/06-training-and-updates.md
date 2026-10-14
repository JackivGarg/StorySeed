# 1.6: watch the scores become learned parameters

We have built inputs, a score table and a loss. Training now joins them in a repeated update. This is the ANN pattern you studied, made inspectable at the level of one selected parameter row. Read [training.py](../../src/storyseed/training.py), [06_training.py](../../lessons/phase01/06_training.py), [the actual training report](../../reports/phase01/06_training.md), then [the exercises](06-practice-and-solutions.md).

## Derive one update before using a full optimizer

Start with a three-choice model whose score table is all zeros. An example selects input row 0 and observes target ID 1. Softmax gives `[1/3,1/3,1/3]`, so cross-entropy is ln(3). From the previous chapter, the selected row's gradient is:

```text
g = [1/3, -2/3, 1/3]
```

The other two rows receive zero gradient because the example did not select them. With ordinary stochastic gradient descent and learning rate η = 0.3:

```text
W_new[0] = W_old[0] - ηg = [-0.1,0.2,-0.1]
```

The target's relative score rises, increasing its probability. The demonstration checks the gradient, row values, unchanged unused rows and lower post-update loss. These checks compare PyTorch with a derivation; they are stronger than simply observing that a script exits.

## Understand the four training operations

`optimizer.zero_grad(set_to_none=True)` clears previously accumulated parameter gradients. PyTorch normally adds new backward contributions to existing gradients, which is useful for deliberate accumulation but incorrect if we forget to clear them between independent updates.

`model(inputs)` computes current scores, and `next_token_loss` evaluates the observed targets. `loss.backward()` calculates gradients through this forward computation into W. It does not itself change W. `optimizer.step()` uses those gradients to change the registered parameters. Repeating only the forward pass or backward pass cannot substitute for an update.

The one-step example uses SGD so the arithmetic matches our hand calculation directly. The longer toy run uses Adam, which maintains moving estimates of gradient and squared-gradient behavior and uses them to adapt parameter updates. Its learning rate still matters, but its update is not simply the SGD formula. Adam is the chosen Phase 1 optimizer, not the later proposed AdamW training system. We will study optimizer internals more deeply when the full model needs them.

## What exactly does one training step contain here?

Our 16 authored training stories supply 240 within-story prediction positions, including endings. We use the transparent full-batch pair representation from 1.3, shape `[1,240]`. Every step evaluates each observed occurrence once, so 400 steps present 96,000 token targets to the loss. Because the same full dataset is reused each step, this corresponds to 400 passes through these toy examples. It is not 96,000 distinct tokens or downloaded stories.

The row packs independent bigram questions; the model does not mix adjacent positions. A contextual model cannot automatically reuse this layout as one continuous attention sequence across stories. This phase prioritizes an exact understandable objective rather than a large stochastic data loader.

Configuration is explicit: 400 steps, learning rate 0.1, seed 17, and `device="auto"`, which selects CUDA if available and otherwise CPU. All inputs and model parameters move to the selected device. We use float32 model parameters and integer token tensors; no mixed precision or accumulation machinery is hidden here.

## Read development results without overstating them

The four separate development stories supply 67 target positions. Evaluation uses `torch.no_grad()` and evaluation mode, computes their loss and restores the previous mode. This bigram has no dropout, so the mode switch does not currently change its formula. Keeping the distinction explicit prepares us for layers that behave differently later.

We record loss before training and after selected update counts. Development targets never participate in updates. They share vocabulary and many simple phrases with training, so lower development loss is useful mechanical evidence rather than a serious generalization claim.

The timer measures the loop, including the recorded evaluations; it does not include package import, all setup or dataset construction. CUDA synchronization makes outstanding device operations finish before timing ends. The reported CUDA peak is allocated tensor memory for this process, not total GPU memory used by the driver or other applications. These definitions matter when interpreting measurements.

## Why the loss cannot reach zero on these examples

The same input token can have several observed successors. Let c_ij count occurrences of successor j after current token i, and n_i be the sum of counts in row i. The best possible bigram distribution on these occurrences is `q(j|i) = c_ij/n_i`. Its corpus-average negative log-probability is the conditional empirical entropy:

```text
H = -sum_(observed i,j) c_ij × ln(q(j|i)) / total_positions
```

Our code computes this reference directly from counts. It does not initialize the model with them; learning still begins from random weights. Finite logits approximate, rather than exactly attain, distributions with zero probability for unobserved transitions. Thus the recorded reference is an infimum, and trained loss should remain at or above it within numerical tolerance.

Learning near this value explains a successful optimization experiment and the architectural ceiling at the same time. The table can learn character-transition frequencies while still failing to maintain a story's plot.

In the actual CUDA run, training loss fell from 2.835582 to 1.212152, and development loss from 2.834067 to 1.216261. The count-derived training reference is 1.210512. After `m`, the training data has 8 occurrences of `i` among 22 successors, giving an empirical probability 8/22, approximately 0.363636. The model learned approximately 0.362914 for that choice. This is evidence that the table approximates observed transition frequencies; it does not establish a coherent story generator.

## Next: use the learned probabilities to produce text

Actual learning is now present. Check the report's first and last loss values, the one-step row update, and the learned successors of `m`. Then explain how this differs from memorizing an entire sentence. The next chapter will feed selected outputs back as inputs to generate a continuation.
