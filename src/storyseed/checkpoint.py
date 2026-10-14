"""Save a small inference checkpoint; complete training resume comes later."""

import hashlib
from pathlib import Path

import torch

from storyseed.bigram import BigramLanguageModel
from storyseed.tokenizer import CharacterTokenizer
from storyseed.training import choose_device

FORMAT = "storyseed.bigram.v1"


def save_checkpoint(path: Path, model, tokenizer: CharacterTokenizer, training: dict) -> dict:
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("Cannot save mismatched model and tokenizer.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "model": {"type": "character_bigram", "vocab_size": tokenizer.vocab_size},
        "tokens": list(tokenizer.tokens),
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "training": training,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"format": FORMAT, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_checkpoint(path: Path, *, device: str = "cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Unsupported checkpoint format.")
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError("Checkpoint vocabulary is invalid.")
    tokenizer = CharacterTokenizer(tuple(tokens))
    configuration = payload.get("model")
    if configuration != {"type": "character_bigram", "vocab_size": tokenizer.vocab_size}:
        raise ValueError("Checkpoint model configuration and vocabulary disagree.")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or set(state) != {"scores"}:
        raise ValueError("Checkpoint parameter names are invalid.")
    scores = state["scores"]
    if not isinstance(scores, torch.Tensor) or scores.shape != (tokenizer.vocab_size, tokenizer.vocab_size):
        raise ValueError("Checkpoint score-table shape is invalid.")
    if scores.dtype != torch.float32 or not torch.isfinite(scores).all():
        raise ValueError("Checkpoint scores must be finite float32 values.")
    model = BigramLanguageModel(tokenizer.vocab_size)
    model.load_state_dict(state, strict=True)
    model.to(choose_device(device)).eval()
    return model, tokenizer, payload.get("training", {})
