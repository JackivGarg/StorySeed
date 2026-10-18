# Observed Phase 6: 03_accumulation

Actual execution; student understanding remains unmeasured.

```json
{
  "command": ".venv/Scripts/python scripts/phase06/03_inspect_accumulation.py",
  "selected_rows": [
    0,
    1258,
    1
  ],
  "microbatch_target_counts": [
    32,
    1,
    32
  ],
  "global_target_denominator": 65,
  "whole_vs_accumulated_max_gradient_difference": 2.9802322387695312e-08,
  "whole_vs_accumulated_max_updated_weight_difference": 3.4226104617118835e-08,
  "wrong_mean_of_microbatch_means_gradient_difference": 0.6494147777557373,
  "control": "CPU fp32, one thread, dropout disabled, same initial weights/rows, clipping inactive at 1e6",
  "normalization": "sum each microbatch token loss / total true targets across the effective batch",
  "final_holdout_accessed": false
}
```
