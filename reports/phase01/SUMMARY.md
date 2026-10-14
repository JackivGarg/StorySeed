# Phase 1 completed implementation: measured summary

This report summarizes actual runs in the fresh StorySeed repository. It is a character bigram learning experiment on authored toy sentences, not a pretrained model, transformer or TinyStories benchmark.

| Item | Observed |
|---|---|
| Python / PyTorch | 3.12.4 / 2.6.0+cu124 |
| Main training device | CUDA, RTX 3050 Laptop GPU |
| Project environment | venv with explicit access to installed system packages |
| Token types / trainable parameters | 17 / 289 |
| Training / development stories | 16 / 4 |
| Training / development target positions | 240 / 67 |
| Main optimization | 400 full-batch Adam steps, learning rate 0.1, seed 17 |
| Training loss | 2.835582 to 1.212152 |
| Development loss | 2.834067 to 1.216261 |
| Count-derived training-loss reference | 1.210512 |
| Saved primary checkpoint | checkpoints/phase01/bigram.pt, 3,686 bytes |
| Save/reload equality | exact parameter/token ordering and configured generation match |
| Fresh-process verification | child exit 0; complete JSON response matched |
| Contract tests | 8 passed |
| CPU command-line reproduction | ran 400 steps, saved a separate artifact, generated successfully |

Timing/memory definitions and precise values are in 06_training.json. Primary checkpoint hash is in 08_checkpoint.json. CPU reproduction outputs are separate files; they do not overwrite the primary CUDA artifact.

Actual sampled text for prompt `mia ` and seed 17: untrained `mia tcrsdwnsahmeddmt`, trained `mia sat m t.`. Top-k=1 produced `mia sat.`. These illustrate local transition learning and its limits; no coherent-story capability is claimed.

All eight subphases have concept chapters and guided practice/solutions, linked in notes/README.md. Student comprehension is unassessed until explanations or applied exercises provide evidence.
