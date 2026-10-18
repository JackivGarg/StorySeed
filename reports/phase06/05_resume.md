# Observed Phase 6: 05_resume

Actual execution; student understanding remains unmeasured.

```json
{
  "command": ".venv/Scripts/python scripts/phase06/05_inspect_resume.py",
  "control": "CPU fp32, one thread, dropout 0.1, identical config and verified data",
  "comparison": "8 uninterrupted successful updates vs 3 + checkpoint + 5 updates in a fresh CLI process",
  "model_weights_bitwise_equal": true,
  "optimizer_state_bitwise_equal": true,
  "torch_rng_state_equal": true,
  "data_generator_state_equal": true,
  "remaining_row_loss_lr_history_equal": true,
  "weights_only_continuation_max_weight_difference": 0.025985196232795715,
  "completed_updates": 8,
  "attempts": 8,
  "consumed_target_tokens": 2013,
  "checkpoint": {
    "path": "C:\\Users\\JACKIV\\OneDrive\\Desktop\\StorySeed\\checkpoints\\phase06\\resumed-after-8.pt",
    "sha256": "9b537e5900ee06b41d2bcfc1e07defd8039d06a121fdfaeba47dcbd26aace1a3",
    "format": "storyseed.decoder.training.v1",
    "completed_updates": 8
  },
  "scope": "same-environment CPU demonstration; cross-device/version bitwise identity is not claimed",
  "final_holdout_accessed": false
}
```
