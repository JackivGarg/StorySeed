"""An ordered finite-context neural predictor, explicitly without attention."""
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from storyseed.token_features import TokenEmbedding
from storyseed.story_batches import DocumentBatches


@dataclass(frozen=True)
class ContextConfig:
    vocabulary: int = 512
    context: int = 8
    dimensions: int = 32
    hidden: int = 64

    def __post_init__(self):
        if self.vocabulary < 3 or min(self.context, self.dimensions, self.hidden) < 1:
            raise ValueError('Model dimensions must be positive and vocabulary must include controls.')

    def to_dict(self):
        return asdict(self)


class FixedWindowLanguageModel(nn.Module):
    def __init__(self, config: ContextConfig):
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config.vocabulary, config.dimensions)
        self.hidden = nn.Linear(config.context * config.dimensions, config.hidden)
        self.output = nn.Linear(config.hidden, config.vocabulary)

    def forward_contexts(self, contexts):
        if contexts.ndim != 2 or contexts.shape[1] != self.config.context:
            raise ValueError('Expected one ordered [N,K] context per target.')
        features = self.embedding(contexts)
        hidden = torch.tanh(self.hidden(features.flatten(start_dim=1)))
        return self.output(hidden)

    def forward(self, input_ids):
        """Return [B,T,V] scores; each row starts with empty context."""
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError('Expected nonempty [B,T] token rows.')
        k = self.config.context
        windows = F.pad(input_ids, (k-1, 0), value=0).unfold(1, k, 1)
        logits = self.forward_contexts(windows.reshape(-1, k))
        return logits.reshape(*input_ids.shape, self.config.vocabulary)


class CausalWindows:
    """Score the same Phase 2 targets, preserving history at storage boundaries."""
    def __init__(self, split, context):
        if context < 1:
            raise ValueError('Positive context required.')
        source = DocumentBatches(split)
        self.ids, self.offsets, self.context = source.ids, source.offsets, context
        self.positions = np.concatenate([
            np.arange(int(a), int(b)-1, dtype=np.int64)
            for a, b in zip(self.offsets[:-1], self.offsets[1:])])
        self.starts = np.repeat(self.offsets[:-1], np.diff(self.offsets)-1)

    def __len__(self):
        return len(self.positions)

    def batch(self, rows):
        rows = np.asarray(rows)
        if (rows.ndim != 1 or rows.size == 0 or not np.issubdtype(rows.dtype, np.integer)
                or np.any(rows < 0) or np.any(rows >= len(self))):
            raise ValueError('Nonempty valid integer target indices required.')
        positions, starts = self.positions[rows], self.starts[rows]
        indices = positions[:, None] + np.arange(1-self.context, 1)
        valid = indices >= starts[:, None]
        safe = np.maximum(indices, starts[:, None])
        contexts = np.where(valid, self.ids[safe], 0).astype(np.int64)
        targets = self.ids[positions+1].astype(np.int64)
        return torch.from_numpy(contexts.copy()), torch.from_numpy(targets.copy())
