"""Explicit attention components, introduced one conceptual operation at a time."""
import torch
from torch import nn


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
