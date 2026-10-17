# Observed Phase 4 result: 04_positions

Actual recorded execution; learner understanding remains unmeasured.

```json
{
  "command": ".venv/Scripts/python scripts/phase04/04_add_positions.py",
  "fixture": "synthetic IDs, seed-17 random token/position features and hand-set QKV projections; not trained language behavior",
  "original_ids": [
    [
      1,
      7,
      8,
      9
    ]
  ],
  "reordered_earlier_ids": [
    [
      1,
      8,
      7,
      9
    ]
  ],
  "without_positions_max_last_output_difference": 0.0,
  "with_positions_max_last_output_difference": 5.9351325035095215e-05,
  "padded_ids": [
    [
      0,
      0,
      1,
      7
    ]
  ],
  "padded_position_ids": [
    [
      0,
      0,
      0,
      1
    ]
  ],
  "unpadded_position_ids": [
    [
      0,
      1
    ]
  ],
  "padding_does_not_shift_real_position_features_max_output_difference": 0.0,
  "padded_states_are_zero": true,
  "position_parameter_shape": [
    8,
    4
  ],
  "position_policy": "0-based real-token offsets inside supplied window; recentered after history truncation",
  "claim_scope": "single attention layer final query; not a claim that all position-free causal stacks are fully order invariant",
  "final_holdout_accessed": false
}
```
