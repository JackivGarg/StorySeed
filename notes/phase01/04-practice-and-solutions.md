# 1.4 practice: understand the lookup before training

1. With V = 5, how many table parameters exist? For input `[B,T]=[2,3]`, what is the score shape?
2. Why do `mia m` and `sam m` produce equal final scores?
3. Is the table's row for `<eos>` automatically a learned stopping rule?

## Worked answers

The table has `5×5=25` parameters. Each of six input positions selects five candidate scores, yielding `[2,3,5]` and 30 output values. Outputs and parameters have different counts because the same parameters are reused.

Both final input IDs identify `m`. The forward calculation selects only that row and does not consult earlier IDs. The tensor has earlier positions, but the architecture decides whether they influence the final calculation.

No. The model learns to predict `<eos>` as a candidate column after observed endings. `<eos>` never appears as an input with an observed successor in our examples, so its input row receives no relevant training. The generation code must stop when that output token is selected.

Try explaining how changing a row of W differs from changing one story's input ID. The first changes shared predictions; the second changes which shared row a particular occurrence selects.
