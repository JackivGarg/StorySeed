# 1.3: arrange prediction questions without changing their meaning

Our text pipeline now knows which token should follow another. A network needs those examples in tensors whose axes have explicit roles. This chapter is about preserving the learning question through that conversion. The implementation is [data.py](../../src/storyseed/data.py); run [03_batches.py](../../lessons/phase01/03_batches.py) and inspect [its report](../../reports/phase01/03_batches.md).

## Recall batching from ANN training

An ANN can process several examples together. A batch is a collection of examples handled by one forward pass; it does not mean their features or labels are interchangeable. For text we name batch size B and the number of input positions in one window T. Here T is a chosen context-window length, not the total story length.

Inputs and targets both have shape `[B,T]`. A value at `[b,t]` is one integer token ID in example b at position t. They are not probabilities or floating-point embeddings. `torch.long` is the integer type we use for lookup indices and target class labels.

## Derive the first window

Use our first story, with its ending marker:

```text
IDs: 8,7,2,0,8,5,13,0,12,2,8,1,16
```

Choose T = 4. At start offset s = 0:

```text
x = ids[0:4] = [8,7,2,0] -> "mia "
y = ids[1:5] = [7,2,0,8] -> "ia m"
```

The four targets are answers for four positions, not a replacement sentence we want to reproduce all at once. At s = 1 the input is `"ia m"` and target is `"a me"`. The two windows overlap; they do not represent independent source stories.

Let N be the encoded length including the ending marker. The final valid start is `N-T-1`, because a T-token input needs T successor targets. With N = 13 and T = 4 there are `N-T = 9` valid starts, 0 through 8. Start 8 gives input `"sam."` and target `"am.<eos>"`. This is why the Python loop uses `range(len(ids)-context)` rather than allowing an additional start that has too few answers.

## Keep a record of where each window came from

`make_windows` iterates through stories first, then offsets within that story. It stores `(document_id,start)` for each row. The demonstration reconstructs every window from that original sequence and checks equality. This directly catches offset and boundary errors; merely checking that two tensors have the same shape would not.

We do not join raw stories before windowing, so a window cannot include one story's ending and the next story's beginning. The input never contains `<eos>` here because there is no observed successor after it. The end marker can appear as the final target.

For this phase, a story too short for the requested window raises a clear error. We deliberately do not silently discard it or invent padding. Later the real data pipeline can use a documented padding/masking policy. The current toy stories support our four-token window.

## Sample a batch and identify what is random

`sample_batch` selects row indices using `torch.randint`. Sampling uses replacement: the same window may appear more than once. A dedicated generator seeded with 17 makes the same row choices when its initial state is recreated. Reusing one generator advances its state, so later calls generally sample different rows. A seed is a starting condition, not a command that every call return the same batch.

If B = 3 and T = 4, the selected input tensor has 12 positions arranged as `[3,4]`. Advanced indexing selects complete rows, and the same indices select targets. Independently choosing target rows would silently destroy supervision even though the shapes still matched.

## Why the bigram trainer will use an additional pair representation

Overlapping windows repeat some prediction positions more often than others. That can change the distribution a simple score table learns. To make its training objective transparent, `transition_pairs` constructs every observed within-story transition once and stores them as one row, `[1,M]`, where M is the total number of character positions across stories.

The row is a convenient list of independent questions for a bigram model, which will use only each current token. It is not a document-packing policy for a transformer. A future contextual model must not treat adjacent pairs from separate documents as continuous context. We retain `make_windows` for learning and later causal layouts while using the flat pairs to inspect the first table's objective exactly.

## What comes next

We have answered how data becomes an ordered, aligned integer tensor. We still cannot predict: IDs alone contain no trainable scores. Next we create a table of floating-point parameters, use an input ID to select a row, and produce scores for the vocabulary choices.

Work [the exercises](03-practice-and-solutions.md) before moving to the score-table chapter. Describe an axis in words as well as giving its dimension; the meaning prevents many bugs that shapes alone cannot detect.
