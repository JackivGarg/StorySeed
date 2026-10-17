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


def causal_attention(query, key, value, valid):
    """Full queries or the final query only; boolean True denotes a real token."""
    if value.ndim != 3 or value.shape[:2] != key.shape[:2]:
        raise ValueError('One value vector per key position required.')
    if valid.dtype != torch.bool or valid.shape != key.shape[:2]:
        raise ValueError('Expected a boolean real-token mask [B,T].')
    length, requests = key.shape[1], query.shape[1]
    if requests not in (1, length):
        raise ValueError('Queries must cover the full sequence or its last position.')
    scores = attention_scores(query, key)
    positions = torch.arange(length, device=scores.device)
    causal = positions[None, :] <= positions[length-requests:, None]
    allowed = causal[None] & valid[:, None, :] & valid[:, -requests:, None]
    has_source = allowed.any(dim=-1, keepdim=True)
    masked = scores.masked_fill(~allowed, float('-inf'))
    # Padded queries may have no source: avoid softmax([-inf,...,-inf]).
    safe = torch.where(has_source, masked, torch.zeros_like(masked))
    weights = torch.where(has_source, safe.softmax(dim=-1), torch.zeros_like(safe))
    return weights @ value, weights


class CausalSelfAttention(nn.Module):
    def __init__(self, dimensions: int, head_dimensions: int):
        super().__init__()
        self.projection = QKVProjection(dimensions, head_dimensions)

    def forward(self, states, valid, *, last_only=False, return_weights=False):
        q, k, v = self.projection(states)
        if last_only:
            q = q[:, -1:, :]
        output, weights = causal_attention(q, k, v, valid)
        return (output, weights) if return_weights else output
