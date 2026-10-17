"""A single explicit causal attention head and vocabulary prediction layer."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from storyseed.position_features import TokenPositionEmbedding
from storyseed.attention import CausalSelfAttention


@dataclass(frozen=True)
class AttentionConfig:
    vocabulary: int = 512
    context: int = 64
    dimensions: int = 32
    head_dimensions: int = 16

    def __post_init__(self):
        values = [self.vocabulary, self.context, self.dimensions, self.head_dimensions]
        if any(type(value) is not int or value < 1 for value in values) or self.vocabulary < 3:
            raise ValueError('Positive integer model dimensions and control-token vocabulary required.')

    def to_dict(self):
        return asdict(self)


class AttentionLanguageModel(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.features = TokenPositionEmbedding(config.vocabulary, config.dimensions, config.context)
        self.attention = CausalSelfAttention(config.dimensions, config.head_dimensions)
        self.output = nn.Linear(config.head_dimensions, config.vocabulary)

    def forward(self, ids):
        states = self.features(ids)
        retrieved = self.attention(states, ids != 0)
        return self.output(retrieved)

    def forward_contexts(self, contexts):
        """One real final query per supplied context; omit unused earlier queries."""
        if contexts.ndim != 2 or contexts.shape[1] < 1 or (contexts[:, -1] == 0).any():
            raise ValueError('Expected [N,T] contexts ending in real current tokens.')
        states = self.features(contexts)
        retrieved = self.attention(states, contexts != 0, last_only=True)
        return self.output(retrieved[:, 0, :])
