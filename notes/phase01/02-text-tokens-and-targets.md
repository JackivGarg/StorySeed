# 1.2: a sentence supplies its own prediction questions

In 1.1, a scalar weight predicted a number and received a gradient. The missing ingredient for language learning is a reliable way to ask many prediction questions. We now build that path from text ourselves. No pretrained tokenizer or model enters this example.

Read [the tokenizer](../../src/storyseed/tokenizer.py), [our authored sentences](../../src/storyseed/toy_data.py), and [02_tokens.py](../../lessons/phase01/02_tokens.py). The observations are in [02_tokens.md](../../reports/phase01/02_tokens.md). Finish with [practice and solutions](02-practice-and-solutions.md).

## Connect this to your LSTM next-word project

Given “mia met”, a next-word model might try to predict “sam”. The basic training structure is familiar: preceding text is the input and the observed continuation supplies the answer. Our first tokenizer chooses a smaller unit, the character. A token is the unit the tokenizer represents; it need not be a word.

For `mia met sam.`, the units are `m`, `i`, `a`, a space, and so on. Spaces and punctuation remain because generating English requires them. One sentence supplies many answers: after `m` the observed character is `i`; after `mi` it is `a`; after `mia` it is a space. These labels come from the text itself. That is self-supervision here. The observed answer is one continuation that occurred, not proof that every other plausible continuation is impossible.

## Build a vocabulary without numerical meanings

We collect distinct training characters, sort them for a stable assignment, and append one special token, `<eos>`, meaning end of story. On our training corpus the ordinary vocabulary is:

```text
0 space, 1 '.', 2 a, 3 c, 4 d, 5 e, 6 h, 7 i,
8 m, 9 n, 10 o, 11 r, 12 s, 13 t, 14 u, 15 w
16 <eos>
```

The vocabulary size V is 17. There are not 17 characters in every sentence: V counts types, while a sentence length counts occurrences. ID 8 is a label for `m`, not a meaningful quantity. Differences between IDs do not measure word meaning or similarity. Later, learned vectors supply numerical representations.

`CharacterTokenizer.fit` is a class method: it constructs a tokenizer from training text. The frozen dataclass stores an immutable tuple of vocabulary items. Validation rejects duplicates and malformed ordinary tokens, and requires the end marker to be last. These constraints make saved configuration interpretable later.

## Encode, then shift

Encoding replaces each character with its ID. Append `<eos>` only when preparing a complete observed training story:

```text
text:       m i a _ m e t _ s a m . <eos>
IDs:        8 7 2 0 8 5 13 0 12 2 8 1 16
inputs:     8 7 2 0 8 5 13 0 12 2 8 1
targets:    7 2 0 8 5 13 0 12 2 8 1 16
```

The underscore above only makes a space visible. We form `inputs = ids[:-1]` and `targets = ids[1:]`. At every aligned position, the target is the next item in the original sequence. There are 12 target positions: the sentence's 12 characters each have a successor once we add the end marker. The final question is “after the full stop, does the story end?”

The later bigram model uses only the current input token. A contextual model can use its permitted prefix; these are different architectures using the same next-token alignment. Even when a training batch stores a complete sentence, a causal model must not read future tokens to answer an earlier position.

We never create a pair from the end marker of one story to the first character of another. Each story is a separate observed sequence. At generation time we do not append `<eos>` to the prompt: the model must predict when to stop.

## Fit only on training data

Our 16 training sentences and four development sentences are original small examples. The development stories are not exact duplicates of training stories, but they deliberately share simple vocabulary and many phrases. This tiny set explains mechanics; it cannot establish serious story quality or broad generalization.

The vocabulary is fitted on training stories only. Development text goes through that fixed mapping. Unknown characters produce a clear error rather than silently receiving new IDs. Encoding `mia!` fails because `!` was not observed in training. This tokenizer does not automatically cover arbitrary English prompts. We will address more useful coverage in Phase 2.

The encode/decode round-trip assertion verifies that supported stories retain their text. Decoding normally hides `<eos>` for readable output; `show_eos=True` can expose the boundary marker for inspection. An invalid ID raises an error, including negative values that would otherwise accidentally index from the end of a Python tuple.

## Why this is a build increment

We now have an explicit tokenizer, a documented toy corpus and inspected input/answer alignment. The tokenizer is reusable implementation code, while the lesson script prints and records its behavior. There are still no model predictions or learning updates in this subphase.

Next we organize these examples into tensors and batches. That step matters because a correct idea can become an incorrect implementation if axes or slice offsets change its meaning.
