# Observed Phase 4 result: 05_training

Actual recorded execution; learner understanding remains unmeasured.

```json
{
  "command": ".venv/Scripts/python scripts/phase04/05_train_attention.py",
  "protocol": {
    "format": "storyseed.phase04.pilot.v1",
    "context": 64,
    "dimensions": 32,
    "head_dimensions": 16,
    "steps": 300,
    "batch_targets": 256,
    "learning_rate": 0.003,
    "initialization_seed": 17,
    "sampling_seed": 2026,
    "device": "cpu",
    "threads": 1,
    "evaluation_batch_targets": 512,
    "compute_deadline_seconds": 45,
    "checkpoint_selection": "fixed final update, no development selection",
    "position_policy": "0-based real offsets within the supplied window, recentered after truncation",
    "training": "uniform target-event sampling with replacement, final query only",
    "scope": "single-seed teaching pilot; no matched-capacity or matched-compute superiority claim"
  },
  "model_configuration": {
    "vocabulary": 512,
    "context": 64,
    "dimensions": 32,
    "head_dimensions": 16
  },
  "stored_parameters_including_fixed_pad_row": 28672,
  "sampled_training_targets_with_replacement": 76800,
  "initial": {
    "train": {
      "nats_per_target_token": 6.246868702689945,
      "perplexity": 516.3933094190379,
      "prediction_positions": 359933
    },
    "development": {
      "nats_per_target_token": 6.246255707060404,
      "perplexity": 516.0768595783334,
      "prediction_positions": 39711
    }
  },
  "final": {
    "train": {
      "nats_per_target_token": 4.651899844165704,
      "perplexity": 104.78386961630271,
      "prediction_positions": 359933
    },
    "development": {
      "nats_per_target_token": 4.653213623932031,
      "perplexity": 104.92162301304108,
      "prediction_positions": 39711
    }
  },
  "minibatch_history": [
    {
      "step": 1,
      "pre_update_minibatch_loss": 6.234622001647949
    },
    {
      "step": 50,
      "pre_update_minibatch_loss": 5.228414535522461
    },
    {
      "step": 100,
      "pre_update_minibatch_loss": 5.102983474731445
    },
    {
      "step": 150,
      "pre_update_minibatch_loss": 4.969006061553955
    },
    {
      "step": 200,
      "pre_update_minibatch_loss": 5.004184246063232
    },
    {
      "step": 250,
      "pre_update_minibatch_loss": 4.7969207763671875
    },
    {
      "step": 300,
      "pre_update_minibatch_loss": 4.7444000244140625
    }
  ],
  "elapsed_compute_seconds": 19.577999999999975,
  "restored_probe_logits_equal": true,
  "checkpoint": {
    "checkpoint": "checkpoints/phase04/attention.pt",
    "format": "storyseed.phase04.attention.v1",
    "sha256": "39ab8a77777d29711f9cb09020426f5a939bbc2a797d5eb146b5a01980f46b4f",
    "tokenizer_sha256": "3540aba1c244ed063b9fb317f0d62280a63d00725cac980e16c3b583cdf549cb",
    "encoded_manifest_sha256": "d16c32659b06511b4d8696976030a1903a17c81a473a8898c9a87f862149c99b",
    "inference_only": true,
    "optimizer_resume": false
  },
  "final_holdout_accessed": false,
  "architecture_scope": "one attention head plus token/position lookup and vocabulary head; no residual, LayerNorm, FFN, dropout or decoder stack",
  "comparison_limits": "same tokenizer/splits/target population; different architecture/capacity/compute from prior baselines"
}
```
