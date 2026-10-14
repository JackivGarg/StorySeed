# 1.8: trained numbers become a reusable model

Our generator has so far trained the tiny model again before producing text. That helps inspect the build, but a useful saved model should generate without retraining. We now preserve the learned table and everything needed to interpret it. Read [checkpoint.py](../../src/storyseed/checkpoint.py), [the save/reload experiment](../../lessons/phase01/08_checkpoint.py), [its report](../../reports/phase01/08_checkpoint.md), and [practice](08-practice-and-solutions.md).

## A tensor alone cannot tell us what it means

Suppose a saved score table contains 289 values. That count suggests a 17-by-17 matrix but does not prove what row 8 means. If the vocabulary ordering changes, the same numbers describe different transitions. The checkpoint therefore includes the exact ordered vocabulary and model configuration, not just weights.

For this bigram, the only registered parameter is `scores`. `model.state_dict()` gives a named mapping from parameter name to tensor. A larger network's state includes multiple parameter tensors and any persistent buffers. The state dictionary is separate from the Python class that defines the forward calculation; loading values does not recreate arbitrary missing source code.

Our schema is `storyseed.bigram.v1`. It identifies the expected format, model type and vocabulary size. The payload also records training configuration and measurements. This metadata describes provenance; it does not participate in a forward pass.

## Follow the save path

`save_checkpoint` checks matching vocabulary sizes and copies state values to CPU using `detach().cpu().clone()`. Detaching removes the training graph, moving to CPU makes storage independent of a required GPU, and cloning preserves a snapshot instead of a reference to a tensor that might change during another update.

The helper writes a temporary file next to the destination and then replaces the destination with the completed file. This reduces the chance of exposing a partially written checkpoint under the final filename. It is a small local inference checkpoint, not a complete crash-recovery training system.

The report includes file size and SHA-256 hash. A hash identifies these exact bytes and helps notice a changed local artifact. It is not an authenticity signature. The checkpoint remains in ignored `checkpoints/phase01/`; Git contains code, notes and the small artifact manifest/report. Another machine can recreate weights by running the documented short training command.

## Reconstruct the model from saved meaning and values

`load_checkpoint` first reads on CPU with the supported weights-only loading mode. Our payload contains tensor state and simple metadata. We validate the schema, token list, unique character vocabulary/end marker, model configuration, parameter names, shape, float32 dtype and finite values. These checks reject a wrong model or a corrupted numerical state instead of silently treating it as a successful load.

Next we instantiate `BigramLanguageModel(V)` and load its named state strictly. Instantiation briefly creates random values, but loading replaces them with trained values; the random initialization is not the source of the restored predictions. The helper then moves the model to the requested available device and enters evaluation mode.

Checkpoint loading does not discover an independently trained tokenizer. It restores the tokenizer ordering saved with these weights. Equal vocabulary sizes alone would not make two different orderings compatible. We verify the exact sequence in the demonstration.

## Verify another process, not merely another variable

The lesson trains and saves a model, checks exact tensor equality after loading, and compares generated text under the same seed. It then launches `scripts/generate.py` in another Python process. That child imports the package and loads the local artifact from disk; it has no access to the parent's model object or computation graph.

The child's complete JSON result must match the parent result. Generation uses CPU probability/sampling operations, so identical saved score values and generator state reproduce this configured example even though training used CUDA. This verifies the supported path on this machine and software version, not a universal cross-version determinism guarantee.

## Reproduce training and then inference yourself

From the repository root:

```powershell
.venv\Scripts\python scripts\train_phase01.py --device auto
.venv\Scripts\python scripts\generate.py --prompt "mia " --seed 17
.venv\Scripts\python scripts\generate.py --prompt "sam " --top-k 1
```

The first command trains from new random initialization under the recorded seed/configuration and saves weights. The second and third load the saved model without training. They support explicit checkpoint paths, output budgets, temperature and device choices. `--json` exposes token IDs and the stopping reason for inspection. Unknown characters still produce a clear error because our toy tokenizer is strict.

## Saving for inference is not resuming training

We store training metadata but do not store optimizer state or the full training data/random-state continuation. Loading this checkpoint restores inference. Starting a new optimizer from its weights would be a new training continuation with different optimizer history, not an exact resume of the old run. Phase 6 will develop a complete resume protocol; the report explicitly marks optimizer resuming unsupported here.

## What Phase 1 has established

We now have our own tokenizer, aligned examples, trainable model, loss, update loop, sampler and reusable saved weights. The model learned local character transitions on authored toy data. We have not implemented the final transformer, downloaded TinyStories, or established coherent story quality.

Review [the final exercises](08-practice-and-solutions.md), especially the distinction between code, weights and token meanings. The next phase replaces toy data with a documented story pipeline while preserving the training task we now understand.
