"""Boundary-aware examples for the first language-model experiments."""

from collections.abc import Sequence

import torch

from storyseed.tokenizer import CharacterTokenizer


def make_windows(stories: Sequence[str], tokenizer: CharacterTokenizer, context: int):
    if context < 1:
        raise ValueError("Context length must be positive.")
    inputs, targets, locations = [], [], []
    for document_id, story in enumerate(stories):
        ids = tokenizer.encode(story, add_eos=True)
        if len(ids) <= context:
            raise ValueError(f"Story {document_id} has too few tokens for context {context}.")
        for start in range(len(ids) - context):
            inputs.append(ids[start:start + context])
            targets.append(ids[start + 1:start + context + 1])
            locations.append((document_id, start))
    if not inputs:
        raise ValueError("No training windows were produced.")
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long), locations


def sample_batch(inputs, targets, batch_size: int, generator: torch.Generator):
    if batch_size < 1 or inputs.ndim != 2 or inputs.shape != targets.shape:
        raise ValueError("Expected matching 2D examples and a positive batch size.")
    if len(inputs) == 0:
        raise ValueError("Cannot sample an empty dataset.")
    rows = torch.randint(len(inputs), (batch_size,), generator=generator)
    return inputs[rows], targets[rows]


def transition_pairs(stories: Sequence[str], tokenizer: CharacterTokenizer):
    """One occurrence of each within-story transition, including period -> EOS.

    The bigram does not mix columns. This flattened layout is not a later
    attention-model document-packing policy.
    """
    inputs, targets = [], []
    for story in stories:
        ids = tokenizer.encode(story, add_eos=True)
        inputs.extend(ids[:-1])
        targets.extend(ids[1:])
    if not inputs:
        raise ValueError("No prediction pairs were produced.")
    return torch.tensor([inputs], dtype=torch.long), torch.tensor([targets], dtype=torch.long)
