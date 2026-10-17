"""Trainable features shared by every occurrence of a token ID."""
import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocabulary: int, dimensions: int, pad_id: int = 0):
        super().__init__()
        if vocabulary < 3 or dimensions < 1 or not 0 <= pad_id < vocabulary:
            raise ValueError('Positive feature dimensions and a valid padding ID are required.')
        self.lookup = nn.Embedding(vocabulary, dimensions, padding_idx=pad_id)

    def forward(self, ids):
        if ids.dtype != torch.long:
            raise ValueError('Embedding inputs must be integer torch.long IDs.')
        return self.lookup(ids)
