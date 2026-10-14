"""Load our own saved bigram weights and generate in a fresh process."""

import argparse
import json
from pathlib import Path

from storyseed.checkpoint import load_checkpoint
from storyseed.generation import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/phase01/bigram.pt"))
    parser.add_argument("--prompt", default="mia ")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        model, tokenizer, _ = load_checkpoint(args.checkpoint, device=args.device)
        result = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                          seed=args.seed, temperature=args.temperature, top_k=args.top_k)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2) if args.json else result["text"])


if __name__ == "__main__":
    main()
