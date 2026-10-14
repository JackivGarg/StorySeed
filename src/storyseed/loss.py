"""Token-weighted next-token error for an unpadded Phase 1 batch."""

import torch
from torch.nn import functional as F


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[:2] != targets.shape:
        raise ValueError("Expected logits [B,T,V] and matching targets [B,T].")
    if targets.dtype != torch.long or targets.numel() == 0:
        raise ValueError("Targets must be nonempty integer class labels.")
    vocabulary = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocabulary), targets.reshape(-1))
