# 1.4: give each current token a row of trainable scores

Our batches contain questions, but no mechanism for predicting their answers. We now add the smallest useful model: a table that assigns next-token scores based on the current token. The implementation is [bigram.py](../../src/storyseed/bigram.py), demonstrated by [04_bigram.py](../../lessons/phase01/04_bigram.py). Read [the observed report](../../reports/phase01/04_bigram.md) and [practice](04-practice-and-solutions.md) alongside this chapter.

## Recall what makes an ANN trainable

In your ANN work, layers contained parameters: numbers whose values changed when a loss was differentiated and an optimizer updated them. A parameter need not belong to a deep stack of layers to be trainable. Our first model consists entirely of one parameter matrix. This removes distractions while preserving the important learning mechanism.

A bigram here models the next token from one current token. The pair of token types gives the name. Our tokens are characters plus an ending marker, so this is a character bigram model. It will not interpret an entire sentence or retain an LSTM-like hidden state.

## Derive the table shape

There are V possible current token IDs and V possible successor IDs. We therefore create a matrix W of shape `[V,V]`. A row identifies the current token; a column identifies a candidate next token. For V = 17, the table has `17 × 17 = 289` scalar parameters.

Write its forward calculation as:

```text
z[b,t,j] = W[x[b,t], j]
```

Here x is the integer input tensor `[B,T]`, b chooses a batch example, t chooses its position, and j ranges across candidate vocabulary items. z is the resulting floating-point score tensor `[B,T,V]`. An input position selects one entire row; it does not multiply the ID as a numerical feature.

For a tiny three-token vocabulary, imagine the row for token `a` is `[0.2,1.0,-0.3]`. If the input is `a`, these three values become its successor scores. Repeating `a` selects the same row again. Changing that row affects every example whose current token is `a`; it does not create a distinct parameter for each story occurrence.

## Scores are not probabilities

The values in z are **logits**, unnormalized real-valued scores. They can be negative or positive, and they need not sum to one. Saying the score for a candidate is larger means the model prefers it relative to the alternatives; its absolute value has no standalone probability meaning.

We initialize W with `torch.randn(V,V) * 0.01`. The multiplication makes the initial random values small. No pretrained weights or observed transition counts are loaded. This random table is the model's untrained starting point. The example's seed reproduces initialization in the same configured software context; it does not make arbitrary devices and future versions universally identical.

`nn.Parameter` registers W so `model.parameters()` exposes it to an optimizer. Subclassing `nn.Module` gives a common forward-call interface and parameter/state handling. `super().__init__()` initializes that machinery. These conveniences do not supply a pretrained language model; we define the actual computation ourselves.

## Trace the real forward pass

Our demonstration encodes two five-character strings, `mia m` and `sam m`. The batch has shape `[2,5]`. `self.scores[token_ids]` uses each integer entry as a row selector, resulting in `[2,5,17]`. The assertion checks an actual row selection against W's `m` row.

Both strings finish in `m`, so their final logits are exactly equal. Earlier letters do not participate in that calculation. This is a useful experiment because it demonstrates the model's limitation directly: a longer input tensor does not automatically mean a model uses longer context.

The forward method rejects non-integer or non-2D input. IDs must be valid vocabulary indices; data preparation guarantees this for normal model input. Bounds validation matters especially at external interfaces, which later tokenization/checkpoint validation supplies. A floating-point `8.0` is not treated as a learned numerical feature for `m`.

## Why the table is a helpful first model

Our corpus contains `m` followed by `i`, `e`, a space, and a period in different places. A single `m` row must represent all those possibilities. There is no one universal successor that can answer all occurrences correctly. Later the loss encourages a probability distribution reflecting the observed ambiguity.

This model is enough to demonstrate training, gradient accumulation and generation, but coherent stories need more context than one character. We will keep that limitation visible rather than equate a functioning model with a good story writer.

No loss is computed and no weights are updated in this subphase. Next we convert relative scores into probabilities and measure how poorly they predict observed target IDs. Then the same parameter registration will let gradients change the table.
