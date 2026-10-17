# Observed Phase 3 result: 01_embeddings

Actual command output; learner understanding remains unmeasured.

```json
{
  "command": ".venv/Scripts/python scripts/phase03/01_inspect_embeddings.py",
  "seed": 17,
  "vocabulary": 512,
  "embedding_dimensions": 32,
  "ids": [
    [
      0,
      1,
      7,
      7,
      9
    ]
  ],
  "input_shape": [
    1,
    5
  ],
  "feature_shape": [
    1,
    5,
    32
  ],
  "parameter_shape": [
    512,
    32
  ],
  "parameter_count": 16384,
  "one_hot_multiplication_matches_lookup": true,
  "first_four_features_of_id_7": [
    0.036597833037376404,
    0.15513989329338074,
    0.9422190189361572,
    0.6066363453865051
  ],
  "loss_for_gradient_probe": "sum of all output features; not the language-model objective",
  "gradient_first_coordinate": {
    "0": 0.0,
    "1": 1.0,
    "7": 2.0,
    "8": 0.0,
    "9": 1.0
  },
  "meaning": "Repeated occurrences share one parameter row; their gradient contributions add.",
  "training_status": "random features; no semantic or language-model learning claim",
  "final_holdout_accessed": false
}
```
