# Observed Phase 6: 06_resources

Actual execution; student understanding remains unmeasured.

```json
{
  "command": ".venv/Scripts/python scripts/phase06/06_measure_resources.py",
  "device": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
  "torch_version": "2.6.0+cu124",
  "measured_configuration": {
    "model": {
      "vocabulary": 512,
      "context": 128,
      "dimensions": 256,
      "heads": 8,
      "layers": 8,
      "multiplier": 4,
      "dropout": 0.1
    },
    "score_tokens": 64,
    "micro_batch_rows": 4,
    "accumulation_steps": 4,
    "total_updates": 2000,
    "warmup_updates": 100,
    "learning_rate": 0.0003,
    "minimum_learning_rate": 3e-05,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "clip_norm": 1.0,
    "seed": 17,
    "data_seed": 2026,
    "device": "cuda",
    "precision": "bf16",
    "threads": 1,
    "evaluation_batch_rows": 8
  },
  "parameter_ledger": {
    "token": 131072,
    "position": 32768,
    "blocks": [
      788736,
      788736,
      788736,
      788736,
      788736,
      788736,
      788736,
      788736
    ],
    "final_norm": 512,
    "vocabulary_head": 131072,
    "total": 6605312
  },
  "warmup_attempts": 2,
  "timed_successful_updates": 3,
  "timed_seconds": 1.4583546000067145,
  "scored_target_tokens": 3008,
  "processed_real_input_tokens": 5824,
  "scored_target_tokens_per_second": 2062.5984928399107,
  "processed_input_tokens_per_second": 3993.541762732593,
  "seconds_per_optimizer_update": 0.48611820000223815,
  "peak_torch_allocated_bytes": 213870592,
  "peak_torch_reserved_bytes": 222298112,
  "device_free_bytes_before_candidate": 5345640448,
  "device_total_bytes": 6441926656,
  "projected_2000_update_compute_seconds": 972.2364000044763,
  "projection_scope": "short stabilized pilot extrapolation; excludes periodic evaluation/checkpoint overhead and thermal variation",
  "cuda_fp16_resume_observation": {
    "saved_scale": 512.0,
    "restored_scale": 512.0,
    "next_rows_equal": true,
    "next_update_max_weight_difference": 0.0,
    "controlled_overflow": {
      "controlled_gradient_overflow_skips_optimizer": true,
      "weights_unchanged": true,
      "successful_update_counter_unchanged": true,
      "loss_scale_before": 1024.0,
      "loss_scale_after": 512.0
    },
    "scope": "same-process CUDA fp16 state-restoration observation after a deliberately skipped attempt"
  },
  "main_training_launched": false,
  "candidate_corpus_scope": "existing teaching prefix only; main corpus/protocol still to freeze in Phase 7",
  "final_holdout_accessed": false
}
```
