"""Subphase 1.4: inspect the parameter table and its context limitation."""

import torch

from storyseed.bigram import BigramLanguageModel
from storyseed.evidence import record
from storyseed.tokenizer import CharacterTokenizer
from storyseed.toy_data import TRAIN_STORIES


def main() -> None:
    torch.manual_seed(17)
    tokenizer = CharacterTokenizer.fit(TRAIN_STORIES)
    model = BigramLanguageModel(tokenizer.vocab_size)
    # These prefixes end in the same token but carry different prior context.
    inputs = torch.tensor([tokenizer.encode("mia m"), tokenizer.encode("sam m")])
    logits = model(inputs)
    assert logits.shape == (2, 5, 17)
    assert torch.equal(logits[0, -1], logits[1, -1])
    assert torch.equal(logits[0, 0], model.scores[8])
    record("04_bigram", {
        "command": ".venv/Scripts/python lessons/phase01/04_bigram.py",
        "seed": 17, "input_shape": list(inputs.shape),
        "output_shape": list(logits.shape),
        "parameter_shape": list(model.scores.shape),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
        "m_row_logits": model.scores[8].detach().tolist(),
        "same_last_token_same_prediction_scores": True,
        "training_performed": False,
    })


if __name__ == "__main__":
    main()
