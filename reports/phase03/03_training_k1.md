# Observed Phase 3 result: 03_training_k1

Actual command output; learner understanding remains unmeasured.

```json
{
  "command": ".venv/Scripts/python scripts/phase03/03_train_context.py --context 1",
  "configuration": {
    "format": "storyseed.phase03.pilot.v1",
    "contexts": [
      1,
      8
    ],
    "dimensions": 32,
    "hidden": 64,
    "steps": 300,
    "batch_targets": 256,
    "learning_rate": 0.003,
    "initialization_seed": 17,
    "sampling_seed": 2026,
    "device": "cpu",
    "threads": 1,
    "evaluation_batch_targets": 2048,
    "compute_deadline_seconds_per_condition": 40,
    "selection": "fixed final update; no development checkpoint selection",
    "scope": "one seed, source-prefix teaching pilot; parameter and compute budgets are not matched to Phase 2"
  },
  "model_configuration": {
    "vocabulary": 512,
    "context": 1,
    "dimensions": 32,
    "hidden": 64
  },
  "stored_parameters_including_fixed_pad_row": 51776,
  "sampled_training_targets_with_replacement": 76800,
  "sampling": "uniform over training target events, with replacement; same index RNG for both conditions",
  "initial": {
    "train": {
      "nats_per_target_token": 6.275427476512456,
      "perplexity": 531.3534743462837,
      "prediction_positions": 359933
    },
    "development": {
      "nats_per_target_token": 6.275551231004437,
      "perplexity": 531.4192357946173,
      "prediction_positions": 39711
    }
  },
  "final": {
    "train": {
      "nats_per_target_token": 3.847273243634513,
      "perplexity": 46.86509914001599,
      "prediction_positions": 359933
    },
    "development": {
      "nats_per_target_token": 3.8677539540720822,
      "perplexity": 47.83482611731429,
      "prediction_positions": 39711
    }
  },
  "minibatch_history": [
    {
      "step": 1,
      "pre_update_minibatch_loss": 6.285745143890381
    },
    {
      "step": 50,
      "pre_update_minibatch_loss": 5.121487140655518
    },
    {
      "step": 100,
      "pre_update_minibatch_loss": 4.541871547698975
    },
    {
      "step": 150,
      "pre_update_minibatch_loss": 4.394965171813965
    },
    {
      "step": 200,
      "pre_update_minibatch_loss": 4.257378578186035
    },
    {
      "step": 250,
      "pre_update_minibatch_loss": 3.928865671157837
    },
    {
      "step": 300,
      "pre_update_minibatch_loss": 3.9435102939605713
    }
  ],
  "elapsed_compute_seconds": 4.5,
  "restored_probe_logits_equal": true,
  "checkpoint": {
    "checkpoint": "checkpoints/phase03/context_k1.pt",
    "sha256": "15c984730e10e963e984a84bb220d668dae5398dd3fa20797a37da0d7f24458b",
    "format": "storyseed.phase03.context.v1",
    "tokenizer_sha256": "3540aba1c244ed063b9fb317f0d62280a63d00725cac980e16c3b583cdf549cb",
    "encoded_manifest_sha256": "d16c32659b06511b4d8696976030a1903a17c81a473a8898c9a87f862149c99b",
    "inference_only": true,
    "optimizer_resume": false
  },
  "final_holdout_accessed": false,
  "limits": "single-seed fixed-budget demonstration; not a matched-parameter or matched-compute architecture study"
}
```
