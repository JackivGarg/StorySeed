# Observed Phase 5: 06_tiny_learning

Actual execution; student understanding remains unmeasured.

```json
{
  "command": ".venv/Scripts/python scripts/phase05/06_learn_tiny_batch.py",
  "configuration": {
    "vocabulary": 512,
    "context": 32,
    "dimensions": 32,
    "heads": 4,
    "layers": 2,
    "multiplier": 4,
    "dropout": 0.0
  },
  "parameter_ledger": {
    "token": 16384,
    "position": 1024,
    "blocks": [
      12576,
      12576
    ],
    "final_norm": 64,
    "vocabulary_head": 16384,
    "total": 59008
  },
  "selected_training_rows": [
    0,
    1,
    17,
    11663
  ],
  "true_targets_per_row": [
    32,
    32,
    32,
    28
  ],
  "initial_tiny_batch_nll": 6.242737770080566,
  "final_tiny_batch_nll": 0.0017903411062434316,
  "updates": 200,
  "learning_rate": 0.01,
  "minibatch_history": [
    {
      "step": 1,
      "pre_update_loss": 6.242737770080566
    },
    {
      "step": 50,
      "pre_update_loss": 0.011310677044093609
    },
    {
      "step": 100,
      "pre_update_loss": 0.00371220032684505
    },
    {
      "step": 150,
      "pre_update_loss": 0.002503481227904558
    },
    {
      "step": 200,
      "pre_update_loss": 0.0018012553919106722
    }
  ],
  "training_compute_seconds": 4.077999999994063,
  "future_input_change_max_earlier_logit_difference": 0.0,
  "earlier_objective_future_state_gradient": 0.0,
  "saved_weights_restore_same_logits": true,
  "fresh_process_sample_matches": true,
  "sample": {
    "prompt": "Once upon a time",
    "text": "Once upon a time, there was a boy called George. He was three years old and one. He wished hes old and ones old and ones",
    "generated_ids": [
      14,
      406,
      283,
      261,
      510,
      278,
      397,
      266,
      223,
      41,
      71,
      303,
      73,
      71,
      16,
      312,
      283,
      307,
      272,
      71,
      331,
      71,
      290,
      85,
      308,
      325,
      267,
      348,
      71,
      16,
      312,
      263,
      289,
      474,
      281,
      85,
      308,
      325,
      267,
      348,
      71,
      85,
      308,
      325,
      267,
      348,
      71,
      85
    ],
    "stop_reason": "length",
    "seed": 17,
    "temperature": 0.8,
    "top_k": 40,
    "context": 32,
    "max_new_tokens": 48,
    "policy": "forbid PAD/BOS while sampling; full vocabulary remains in evaluation loss"
  },
  "checkpoint": {
    "path": "C:\\Users\\JACKIV\\OneDrive\\Desktop\\StorySeed\\checkpoints\\phase05\\tiny-decoder.pt",
    "sha256": "b810810c520edae096d610537b53673e0eaa7d363c237a27714d7100a9938092",
    "format": "storyseed.decoder.inference.v1",
    "purpose": "four fixed training chunks: debugging overfit, not corpus quality",
    "optimizer_resume": false
  },
  "scope": "deliberate memorization of four training chunks, not development/final story quality",
  "development_scored": false,
  "final_holdout_accessed": false
}
```
