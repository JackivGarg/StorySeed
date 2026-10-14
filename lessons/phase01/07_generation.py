"""Subphase 1.7: sample before/after training and expose missing context."""

import torch

from storyseed.bigram import BigramLanguageModel
from storyseed.evidence import record
from storyseed.generation import generate
from storyseed.training import train_bigram


def main() -> None:
    trained, tokenizer, training = train_bigram()
    torch.manual_seed(17)
    untrained = BigramLanguageModel(tokenizer.vocab_size)
    before = generate(untrained, tokenizer, "mia ", seed=17)
    after = generate(trained, tokenizer, "mia ", seed=17)
    assert after == generate(trained, tokenizer, "mia ", seed=17)
    first = generate(trained, tokenizer, "mia m", seed=17)
    second = generate(trained, tokenizer, "sam m", seed=17)
    assert first["generated_token_ids"] == second["generated_token_ids"]
    zero = generate(trained, tokenizer, "mia ", max_new_tokens=0)
    assert zero["text"] == "mia " and zero["sampled_tokens"] == 0
    record("07_generation", {
        "command": ".venv/Scripts/python lessons/phase01/07_generation.py",
        "training_policy": "retrain the same small configured toy model; checkpoint loading is introduced next",
        "training_device": training["actual_device"], "before_training": before,
        "after_training": after,
        "other_seed": generate(trained, tokenizer, "mia ", seed=18),
        "top_k_one": generate(trained, tokenizer, "mia ", top_k=1, max_new_tokens=40),
        "same_final_token_different_prefixes": [first, second],
        "identical_sample_on_repeated_seed": True,
        "zero_generation_budget": zero,
    })


if __name__ == "__main__":
    main()
