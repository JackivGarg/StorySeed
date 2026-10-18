# Observed Phase 6: 04_precision

Actual execution; student understanding remains unmeasured.

```json
{
  "command": ".venv/Scripts/python scripts/phase06/04_inspect_precision.py",
  "torch_version": "2.6.0+cu124",
  "cuda_available": true,
  "device": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
  "observations": {
    "fp32": {
      "forward_logits_dtype": "torch.float32",
      "initial_nll": 6.256446361541748,
      "absolute_nll_difference_from_fp32": 0.0,
      "one_update_seconds": 0.5087304000044242,
      "optimizer_step_skipped": false,
      "unscaled_gradient_norm": 1.0862094163894653,
      "loss_scale": 1.0,
      "peak_allocated_bytes": 20651520,
      "scope": "single small workload including first optimizer-state allocation; not a stabilized throughput benchmark"
    },
    "fp16": {
      "forward_logits_dtype": "torch.float16",
      "initial_nll": 6.256451606750488,
      "absolute_nll_difference_from_fp32": 5.245208740234375e-06,
      "one_update_seconds": 0.09269609999319073,
      "optimizer_step_skipped": false,
      "unscaled_gradient_norm": 1.086201786994934,
      "loss_scale": 1024.0,
      "peak_allocated_bytes": 20191232,
      "scope": "single small workload including first optimizer-state allocation; not a stabilized throughput benchmark"
    },
    "bf16": {
      "forward_logits_dtype": "torch.bfloat16",
      "initial_nll": 6.256457805633545,
      "absolute_nll_difference_from_fp32": 1.1444091796875e-05,
      "one_update_seconds": 0.054640899994410574,
      "optimizer_step_skipped": false,
      "unscaled_gradient_norm": 1.0869642496109009,
      "loss_scale": 1.0,
      "peak_allocated_bytes": 20191232,
      "scope": "single small workload including first optimizer-state allocation; not a stabilized throughput benchmark"
    }
  },
  "policy": "fp32 master parameters; autocast selected CUDA operations; CE in fp32; fp16 scaler unscaled once after accumulation",
  "limits": "finite one-update evidence does not establish long-run numeric stability or quality",
  "final_holdout_accessed": false
}
```
