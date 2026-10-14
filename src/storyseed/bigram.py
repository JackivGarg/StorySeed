"""A trainable table: one current token supplies next-token scores."""

import torch
from torch import nn


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        if vocab_size < 2:
            raise ValueError("Vocabulary must contain at least two tokens.")
        self.vocab_size = vocab_size
        self.scores = nn.Parameter(torch.randn(vocab_size, vocab_size) * 0.01)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("Expected a torch.long token tensor of shape [B,T].")
        return self.scores[token_ids]
