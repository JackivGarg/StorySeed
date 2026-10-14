"""Subphase 1.2: inspect categorical IDs and shifted observed answers."""

from storyseed.evidence import record
from storyseed.tokenizer import CharacterTokenizer
from storyseed.toy_data import DEV_STORIES, TRAIN_STORIES


def main() -> None:
    assert not set(TRAIN_STORIES).intersection(DEV_STORIES)
    tokenizer = CharacterTokenizer.fit(TRAIN_STORIES)
    for story in TRAIN_STORIES + DEV_STORIES:
        assert tokenizer.decode(tokenizer.encode(story, add_eos=True)) == story
    ids = tokenizer.encode(TRAIN_STORIES[0], add_eos=True)
    inputs, targets = ids[:-1], ids[1:]
    try:
        tokenizer.encode("mia!")
    except ValueError as error:
        unknown_behavior = str(error)
    else:
        raise AssertionError("Unknown text must not silently change vocabulary.")
    record("02_tokens", {
        "command": ".venv/Scripts/python lessons/phase01/02_tokens.py",
        "corpus": "original authored toy sentences, not TinyStories",
        "train_stories": len(TRAIN_STORIES), "development_stories": len(DEV_STORIES),
        "vocabulary": list(tokenizer.tokens), "vocab_size": tokenizer.vocab_size,
        "example": TRAIN_STORIES[0], "ids_with_eos": ids,
        "inputs": inputs, "targets": targets,
        "prediction_positions": len(targets),
        "last_transition": [tokenizer.tokens[inputs[-1]], tokenizer.tokens[targets[-1]]],
        "round_trip_all_stories": True, "exact_story_overlap": False,
        "unknown_character_behavior": unknown_behavior,
    })


if __name__ == "__main__":
    main()
