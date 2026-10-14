# 1.2 practice: IDs are labels and targets are successors

1. Encode `sam` without an end marker. Then give inputs and targets with and without `<eos>`.
2. Why should a prompt not receive an end-of-story marker before generation?
3. What would training with identical inputs and targets encourage?

## Worked answers

`sam` encodes as `[12,2,8]`. Without an end marker, inputs `[12,2]` have targets `[2,8]`: two predictions. With the marker, the observed sequence is `[12,2,8,16]`, so inputs `[12,2,8]` have targets `[2,8,16]`: three predictions. The extra position teaches an ending; it does not create a fourth observed character.

A generation prompt is unfinished text. Appending the end marker would incorrectly present it as already ended; the generation routine starts with its final ordinary token and allows the model to predict the next token, potentially the end marker.

Identical lists would reward predicting the current token. That teaches copying rather than predicting a successor. Our shifted labels maintain the intended learning question at every position.

Explain why a vocabulary with 17 types can encode a 100-token sequence. Types can recur many times. The number of possible labels and the length of one example are different quantities.
