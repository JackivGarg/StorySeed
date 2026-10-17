"""Learned positions relative to the beginning of the supplied real window."""
import torch
from torch import nn
from storyseed.token_features import TokenEmbedding


class TokenPositionEmbedding(nn.Module):
    def __init__(self, vocabulary, dimensions, max_context):
        super().__init__()
        if type(max_context) is not int or max_context < 1:
            raise ValueError('Positive integer maximum context required.')
        self.max_context = max_context
        self.token = TokenEmbedding(vocabulary, dimensions)
        self.position = nn.Embedding(max_context, dimensions)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)

    def position_ids(self, ids):
        if ids.ndim != 2 or ids.shape[1] < 1 or ids.shape[1] > self.max_context:
            raise ValueError('Expected [B,T] IDs with 1 <= T <= max_context.')
        valid = ids != 0
        return (valid.long().cumsum(dim=-1)-1).clamp_min(0)

    def forward(self, ids):
        positions = self.position_ids(ids)
        states = self.token(ids) + self.position(positions)
        return states * (ids != 0).unsqueeze(-1)
