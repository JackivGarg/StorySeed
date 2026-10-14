"""Subphase 1.5: check a hand-derived probability, loss and score gradient."""

import math

import torch
from torch.nn import functional as F

from storyseed.evidence import record
from storyseed.loss import next_token_loss


def main() -> None:
    logits = torch.tensor([[[math.log(2), 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0]])
    probabilities = logits.softmax(dim=-1)
    loss = next_token_loss(logits, target)
    loss.backward()
    assert torch.allclose(probabilities, torch.tensor([[[.5, .25, .25]]], dtype=torch.float64))
    assert math.isclose(loss.item(), math.log(2), abs_tol=1e-12)
    assert torch.allclose(logits.grad, torch.tensor([[[-.5, .25, .25]]], dtype=torch.float64))
    stronger = torch.tensor([[[math.log(4), 0.0, 0.0]]], dtype=torch.float64)
    stronger_loss = next_token_loss(stronger, target)
    assert stronger_loss < loss
    torch.manual_seed(17)
    batch_logits = torch.randn(2, 3, 17)
    batch_targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
    combined = next_token_loss(batch_logits, batch_targets)
    separate = torch.stack([
        F.cross_entropy(batch_logits[b, t].unsqueeze(0), batch_targets[b, t].unsqueeze(0))
        for b in range(2) for t in range(3)
    ]).mean()
    assert torch.allclose(combined, separate)
    record("05_loss", {
        "command": ".venv/Scripts/python lessons/phase01/05_loss.py",
        "three_choice_logits": logits.detach().tolist(),
        "probabilities": probabilities.detach().tolist(), "target_id": 0,
        "cross_entropy": loss.item(), "hand_expected_loss": math.log(2),
        "score_gradients": logits.grad.tolist(),
        "loss_after_relative_target_score_increase": stronger_loss.item(),
        "incorrect_probability_as_logit_loss": next_token_loss(probabilities.detach(), target).item(),
        "batch_loss": combined.item(), "separate_position_mean": separate.item(),
        "training_performed": False,
    })


if __name__ == "__main__":
    main()
