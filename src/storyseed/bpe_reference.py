"""Small explicit byte-pair learner; independent of the main tokenizer backend."""
from collections import Counter
from dataclasses import dataclass

from storyseed.corpus import record


def merge_pair(sequence, pair, new_id):
    output = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and tuple(sequence[index:index + 2]) == pair:
            output.append(new_id)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return output


@dataclass
class ReferenceBPE:
    merges: list[tuple[int, int]]
    pieces: list[bytes]
    trace: list[dict]

    @classmethod
    def fit(cls, texts: list[str], steps: int = 8):
        if not texts or steps < 0:
            raise ValueError('Nonempty training text and nonnegative steps required.')
        sequences = [list(text.encode('utf-8')) for text in texts]
        pieces = [bytes([i]) for i in range(256)]
        merges, trace = [], []
        for _ in range(steps):
            counts = Counter(pair for ids in sequences for pair in zip(ids, ids[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda candidate: (-counts[candidate], candidate))
            new_id = len(pieces)
            before = sum(map(len, sequences))
            sequences = [merge_pair(ids, pair, new_id) for ids in sequences]
            pieces.append(pieces[pair[0]] + pieces[pair[1]])
            merges.append(pair)
            trace.append({'new_id': new_id, 'pair': list(pair), 'adjacent_count': counts[pair],
                          'piece': pieces[-1].decode('utf-8', errors='backslashreplace'),
                          'tokens_before': before, 'tokens_after': sum(map(len, sequences))})
        return cls(merges, pieces, trace)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode('utf-8'))
        for offset, pair in enumerate(self.merges):
            ids = merge_pair(ids, pair, 256 + offset)
        return ids

    def decode(self, ids: list[int]) -> str:
        if any(type(i) is not int or not 0 <= i < len(self.pieces) for i in ids):
            raise ValueError('Invalid BPE token ID.')
        return b''.join(self.pieces[i] for i in ids).decode('utf-8')


def demonstrate_bpe() -> None:
    examples = ['banana banana', 'banana bandana', 'bandana']
    tokenizer = ReferenceBPE.fit(examples, steps=8)
    probes = examples + ['Mia saw a cat.', 'café 🐈']
    for text in probes:
        assert tokenizer.decode(tokenizer.encode(text)) == text
    record('03_bpe', {'command': '.venv/Scripts/python scripts/phase02/03_bpe_demo.py',
        'fixture': examples, 'base_byte_types': 256, 'merges': tokenizer.trace,
        'encoded_first': tokenizer.encode(examples[0]), 'roundtrip_probes': len(probes),
        'overlap_example': {'input': 'aaaa', 'adjacent_aa_pairs': 3,
                            'nonoverlapping_replacements': len(merge_pair([97]*4, (97,97), 256))},
        'role': 'transparent BPE concept fixture; not the trained corpus tokenizer',
        'tie_break': 'highest adjacent frequency, then lexicographically smallest token-ID pair',
        'boundaries': 'no cross-document merges; unlike main tokenizer, no regex pre-tokenization'})
