"""Subphase 1.6: inspect one exact SGD step, then train the toy bigram."""

import torch

from storyseed.bigram import BigramLanguageModel
from storyseed.evidence import record
from storyseed.loss import next_token_loss
from storyseed.training import train_bigram


def inspect_one_step():
    model = BigramLanguageModel(3)
    with torch.no_grad():
        model.scores.zero_()
    inputs, targets = torch.tensor([[0]]), torch.tensor([[1]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    before = model.scores.detach().clone()
    loss = next_token_loss(model(inputs), targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = model.scores.grad.detach().clone()
    expected = torch.tensor([1 / 3, -2 / 3, 1 / 3])
    assert torch.allclose(gradient[0], expected)
    assert torch.count_nonzero(gradient[1:]).item() == 0
    optimizer.step()
    after_loss = next_token_loss(model(inputs), targets).item()
    assert after_loss < loss.item()
    assert torch.allclose(model.scores[0], torch.tensor([-0.1, 0.2, -0.1]))
    return {"learning_rate": .3, "before_row": before[0].tolist(), "gradient_row": gradient[0].tolist(), "after_row": model.scores[0].detach().tolist(), "loss_before": loss.item(), "loss_after": after_loss, "unvisited_rows_unchanged": torch.equal(model.scores[1:], before[1:])}


def main() -> None:
    one_step = inspect_one_step()
    model, tokenizer, training = train_bigram()
    first, last = training["history"][0], training["history"][-1]
    assert last["train_loss"] < first["train_loss"]
    assert last["train_loss"] >= training["empirical_bigram_train_loss_infimum"] - 1e-5
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    record("06_training", {
        "command": ".venv/Scripts/python lessons/phase01/06_training.py",
        "single_sgd_update": one_step, "main_training": training,
        "vocabulary": list(tokenizer.tokens), "training_performed": True,
    })


if __name__ == "__main__":
    main()
