"""A strict character tokenizer with one explicit end-of-story token."""

from dataclasses import dataclass
from typing import Iterable

EOS = "<eos>"


@dataclass(frozen=True)
class CharacterTokenizer:
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tokens or self.tokens[-1] != EOS:
            raise ValueError("The final vocabulary item must be <eos>.")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocabulary items must be unique.")
        if any(len(token) != 1 for token in self.tokens[:-1]):
            raise ValueError("Ordinary vocabulary tokens must be single characters.")

    @classmethod
    def fit(cls, stories: Iterable[str]) -> "CharacterTokenizer":
        characters = sorted(set("".join(stories)))
        if not characters:
            raise ValueError("Cannot fit a tokenizer on empty text.")
        return cls(tuple(characters) + (EOS,))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def eos_id(self) -> int:
        return len(self.tokens) - 1

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        mapping = {token: index for index, token in enumerate(self.tokens)}
        try:
            ids = [mapping[character] for character in text]
        except KeyError as error:
            raise ValueError(f"Unknown character: {error.args[0]!r}") from error
        return ids + [self.eos_id] if add_eos else ids

    def decode(self, ids: Iterable[int], *, show_eos: bool = False) -> str:
        result = []
        for index in ids:
            if not isinstance(index, int) or not 0 <= index < self.vocab_size:
                raise ValueError(f"Invalid token ID: {index!r}")
            if index != self.eos_id or show_eos:
                result.append(self.tokens[index])
        return "".join(result)
