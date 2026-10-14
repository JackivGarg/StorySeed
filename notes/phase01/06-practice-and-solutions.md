# 1.6 practice: where learning happens

1. Which call changes the parameters: forward, backward, or optimizer.step?
2. Starting from zero logits with target 1 and η = 0.6, compute the SGD-updated row.
3. Why can a correctly trained bigram retain a positive loss?

## Worked answers

The optimizer step changes the parameters. Forward computes predictions; backward computes gradient contributions. In ordinary training we clear accumulated gradients before each new update. Deliberate accumulation requires a different documented schedule.

The gradient remains `[1/3,-2/3,1/3]` for this starting point, so the row becomes `[-0.2,0.4,-0.2]`. This result concerns SGD; Adam's maintained state gives a different update rule.

A current character can have several observed successors. The best one-row distribution must assign probability to them, so each individual occurrence generally receives probability below one and therefore positive negative-log loss. More iterations cannot provide the missing preceding context.

Explain why 400 full-batch steps over 240 target positions expose 96,000 targets but do not create 96,000 independent training examples. Repetition of data and additional information are different things.
