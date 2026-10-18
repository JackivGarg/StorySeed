# Observed Phase 6: 01_run_contract

Actual execution; student understanding remains unmeasured.

```json
{
  "command": ".venv/Scripts/python scripts/phase06/01_define_run.py",
  "configuration": {
    "model": {
      "vocabulary": 512,
      "context": 64,
      "dimensions": 32,
      "heads": 4,
      "layers": 2,
      "multiplier": 4,
      "dropout": 0.1
    },
    "score_tokens": 32,
    "micro_batch_rows": 4,
    "accumulation_steps": 2,
    "total_updates": 20,
    "warmup_updates": 4,
    "learning_rate": 0.003,
    "minimum_learning_rate": 0.0003,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "clip_norm": 1.0,
    "seed": 17,
    "data_seed": 2026,
    "device": "cpu",
    "precision": "fp32",
    "threads": 1,
    "evaluation_batch_rows": 16
  },
  "data_identity": {
    "tokenizer_sha256": "3540aba1c244ed063b9fb317f0d62280a63d00725cac980e16c3b583cdf549cb",
    "encoded_manifest_sha256": "d16c32659b06511b4d8696976030a1903a17c81a473a8898c9a87f862149c99b"
  },
  "training_target_population": 359933,
  "development": {
    "nats_per_target_token": 6.246756208669128,
    "perplexity": 516.3352215266696,
    "prediction_positions": 39711,
    "batches": 81,
    "complete_population": true,
    "precision": "fp32 logits/loss, float64 sum"
  },
  "selected_rows": [
    0,
    1,
    11663
  ],
  "real_input_counts": [
    32,
    64,
    60
  ],
  "scored_target_counts": [
    32,
    32,
    28
  ],
  "evaluation_restored_training_mode": true,
  "context_policy": "up to context-score_tokens warm-up inputs before each disjoint scored block; window-relative positions",
  "final_holdout_accessed": false
}
```
