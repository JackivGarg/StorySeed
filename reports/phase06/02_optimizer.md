# Observed Phase 6: 02_optimizer

Actual execution; student understanding remains unmeasured.

```json
{
  "command": ".venv/Scripts/python scripts/phase06/02_inspect_optimizer.py",
  "optimizer": "AdamW foreach=False; matrices including token/position tables decay, biases/LayerNorm vectors do not",
  "decay_parameter_count": 59392,
  "no_decay_parameter_count": 640,
  "schedule_lr_at_updates": {
    "1": 0.00075,
    "4": 0.003,
    "5": 0.002974060128544361,
    "20": 0.0003
  },
  "first_update_learning_rate": 0.00075,
  "gradient_norm_before_clip": 1.5182970762252808,
  "gradient_norm_after_clip": 0.9999994896866982,
  "output_weight_max_change": 0.0007563233375549316,
  "step_index_semantics": "one-based successful optimizer updates; warmup/cosine is a pure function of config and count",
  "final_holdout_accessed": false
}
```
