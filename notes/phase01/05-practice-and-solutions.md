# 1.5 practice: probabilities, losses and derivatives

1. Two equal logits give what probabilities and target loss?
2. For probabilities `[0.1,0.7,0.2]` and target ID 1, give the loss and score-gradient vector.
3. Why does `[B,T,V]` become `[B*T,V]` rather than `[B,V*T]`?

## Worked answers

Equal logits give `[0.5,0.5]`, with negative log-probability ln(2) for either target. Their absolute value does not matter if they remain equal.

The target probability is 0.7, so loss is `-ln(0.7) ≈ 0.356675`. The derivative is `[0.1,-0.3,0.2]`. Subtracting a small multiple raises the second candidate's relative score. For a mean across N positions, each position's gradient contribution also receives a factor `1/N`.

We have B*T separate prediction questions, each with V possible answers. Combining time and vocabulary would instead create V*T candidates for each batch example, losing the meaning of one next-token question per position.

Checkpoint: explain why lowering token loss alone does not prove a model can maintain a character's identity across a story.
