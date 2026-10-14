"""Subphase 1.3: trace document boundaries, windows, shapes and sampling."""

import torch

from storyseed.data import make_windows, sample_batch, transition_pairs
from storyseed.evidence import record
from storyseed.tokenizer import CharacterTokenizer
from storyseed.toy_data import TRAIN_STORIES


def main() -> None:
    tokenizer = CharacterTokenizer.fit(TRAIN_STORIES)
    x, y, locations = make_windows(TRAIN_STORIES, tokenizer, context=4)
    for row, (document, start) in enumerate(locations):
        ids = tokenizer.encode(TRAIN_STORIES[document], add_eos=True)
        assert x[row].tolist() == ids[start:start + 4]
        assert y[row].tolist() == ids[start + 1:start + 5]
    assert x[0].tolist() == [8, 7, 2, 0]
    assert y[0].tolist() == [7, 2, 0, 8]
    batch_x, batch_y = sample_batch(x, y, 3, torch.Generator().manual_seed(17))
    again_x, again_y = sample_batch(x, y, 3, torch.Generator().manual_seed(17))
    assert torch.equal(batch_x, again_x) and torch.equal(batch_y, again_y)
    pairs_x, pairs_y = transition_pairs(TRAIN_STORIES, tokenizer)
    assert pairs_x.numel() == sum(len(story) for story in TRAIN_STORIES)
    assert (pairs_y == tokenizer.eos_id).sum().item() == len(TRAIN_STORIES)
    try:
        make_windows(["a"], tokenizer, 4)
    except ValueError as error:
        short_story = str(error)
    record("03_batches", {
        "command": ".venv/Scripts/python lessons/phase01/03_batches.py",
        "window_shape": list(x.shape), "dtype": str(x.dtype),
        "batch_shape": list(batch_x.shape), "first_window_x": x[0].tolist(),
        "first_window_y": y[0].tolist(), "locations_checked": len(locations),
        "sample_inputs": [tokenizer.decode(row.tolist(), show_eos=True) for row in batch_x],
        "sample_targets": [tokenizer.decode(row.tolist(), show_eos=True) for row in batch_y],
        "repeatable_seeded_batch": True, "unique_transition_positions": pairs_x.numel(),
        "observed_story_endings": (pairs_y == tokenizer.eos_id).sum().item(),
        "short_story_policy": short_story,
    })


if __name__ == "__main__":
    main()
