# 1.3 practice: alignment, axes and boundaries

1. For an encoded sequence of length 10 and T = 3, how many complete shifted windows exist?
2. For B = 5 and T = 4, what do the two axes mean and how many target positions exist?
3. Why must we select the same sampled rows for inputs and targets?

## Worked answers

There are `10-3=7` starts, from 0 through 6. At start 6, the input uses positions 6,7,8 and the targets use 7,8,9. A start of 7 would require a target at position 10, outside the sequence.

The first axis identifies the five selected windows; the second identifies four aligned positions per window. There are 20 supervised positions. Vocabulary size is another quantity; it is not an input axis yet.

Different row choices could pair a prefix from one part of a story with unrelated answers from another. The arrays would still look numerically valid, so an alignment check is essential.

For a final check, explain why flattening independent bigram pairs is permissible here but cannot simply be reused as one attention context across different stories.
