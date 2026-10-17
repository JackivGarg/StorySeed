"""Explicit attention components, introduced one conceptual operation at a time."""
import torch
from torch import nn
import math


class QKVProjection(nn.Module):
    """Shared learned functions for requesting, matching and carrying information."""
    def __init__(self, dimensions: int, head_dimensions: int):
        super().__init__()
        if min(dimensions, head_dimensions) < 1:
            raise ValueError('Positive input and head dimensions required.')
        self.query = nn.Linear(dimensions, head_dimensions, bias=False)
        self.key = nn.Linear(dimensions, head_dimensions, bias=False)
        self.value = nn.Linear(dimensions, head_dimensions, bias=False)

    def forward(self, states):
        if states.ndim != 3:
            raise ValueError('Expected floating features [B,T,D].')
        return self.query(states), self.key(states), self.value(states)


def attention_scores(query, key):
    if (query.ndim != 3 or key.ndim != 3 or query.shape[0] != key.shape[0]
            or query.shape[-1] != key.shape[-1] or query.shape[-1] < 1
            or key.shape[1] < 1):
        raise ValueError('Expected compatible queries [B,Q,d] and keys [B,T,d].')
    return (query @ key.transpose(-2, -1)) / math.sqrt(query.shape[-1])


def scaled_attention(query, key, value):
    """Unmasked illustration; never use this path for causal model training."""
    if value.ndim != 3 or value.shape[:2] != key.shape[:2]:
        raise ValueError('One value vector per key position required.')
    weights = attention_scores(query, key).softmax(dim=-1)
    return weights @ value, weights
