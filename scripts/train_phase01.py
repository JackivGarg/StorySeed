"""Reproduce the small scratch-training run and save its inference checkpoint."""

import argparse
import json
from pathlib import Path

from storyseed.checkpoint import save_checkpoint
from storyseed.training import TrainingConfig, train_bigram


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/phase01/bigram.pt"))
    args = parser.parse_args()
    model, tokenizer, result = train_bigram(TrainingConfig(args.steps, args.learning_rate, args.seed, args.device))
    artifact = save_checkpoint(args.checkpoint, model, tokenizer, result)
    print(json.dumps({"checkpoint": str(args.checkpoint), "artifact": artifact,
                      "device": result["actual_device"], "final_losses": result["history"][-1]}, indent=2))


if __name__ == "__main__":
    main()
