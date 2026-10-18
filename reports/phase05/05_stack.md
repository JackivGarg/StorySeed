# Observed Phase 5: 05_stack

Actual execution; student understanding remains unmeasured.

```json
{
  "command": ".venv/Scripts/python scripts/phase05/05_inspect_stack.py",
  "configuration": {
    "vocabulary": 512,
    "context": 64,
    "dimensions": 32,
    "heads": 4,
    "layers": 2,
    "multiplier": 4,
    "dropout": 0.1
  },
  "ids": [
    [
      1,
      47,
      75,
      67,
      274,
      388
    ]
  ],
  "feature_and_block_shapes": [
    [
      1,
      6,
      32
    ],
    [
      1,
      6,
      32
    ],
    [
      1,
      6,
      32
    ]
  ],
  "logits_shape": [
    1,
    6,
    512
  ],
  "parameter_ledger": {
    "token": 16384,
    "position": 2048,
    "blocks": [
      12576,
      12576
    ],
    "final_norm": 64,
    "vocabulary_head": 16384,
    "total": 60032
  },
  "token_and_output_weights_are_separate": true,
  "initialization": "normal std 0.02 for linear/embedding weights, zero linear biases, fixed zero PAD row, default LayerNorm affine",
  "next_need": "demonstrate tiny-batch learning and full-stack causal information boundaries",
  "final_holdout_accessed": false
}
```
