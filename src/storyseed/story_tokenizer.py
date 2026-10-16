"""A train-only byte-level BPE tokenizer with explicit control-token policy."""
from importlib.metadata import version
import json

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from storyseed.corpus import ROOT, MANIFESTS, write_json, sha256, record
from storyseed.story_splits import load_split

SPECIALS = ['<|pad|>', '<|bos|>', '<|eos|>']
TOKENIZER_PATH = ROOT / 'artifacts/phase02/tokenizer.json'


class StoryTokenizer:
    pad_id, bos_id, eos_id = 0, 1, 2

    def __init__(self, backend):
        self.backend = backend
        if [backend.token_to_id(t) for t in SPECIALS] != [0, 1, 2]:
            raise ValueError('Tokenizer control IDs differ from the phase contract.')
        self.vocab_size = backend.get_vocab_size()

    def encode(self, text: str, *, add_bos=False, add_eos=False) -> list[int]:
        if any(token in text for token in SPECIALS):
            raise ValueError('Literal control-token strings are reserved, not ordinary input text.')
        ids = self.backend.encode(text, add_special_tokens=False).ids
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids) -> str:
        values = list(ids)
        if any(type(i) is not int or not 0 <= i < self.vocab_size for i in values):
            raise ValueError('Invalid token ID.')
        return self.backend.decode(values, skip_special_tokens=True)

    @classmethod
    def load(cls, path=TOKENIZER_PATH, *, expected_sha256=None):
        if expected_sha256 is None:
            metadata = json.loads((MANIFESTS / 'tokenizer.json').read_text(encoding='utf-8'))
            expected_sha256 = metadata['artifact_sha256']
        if sha256(path.read_bytes()) != expected_sha256:
            raise ValueError('Tokenizer bytes differ from the expected artifact.')
        return cls(Tokenizer.from_file(str(path)))


def fit_story_tokenizer():
    training = load_split('train')
    development = load_split('development')
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=512, min_frequency=2, show_progress=False,
        special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    backend.train_from_iterator((row['text'] for row in training), trainer=trainer, length=len(training))
    tokenizer = StoryTokenizer(backend)
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    backend.save(str(TOKENIZER_PATH))
    digest = sha256(TOKENIZER_PATH.read_bytes())
    restored = StoryTokenizer.load(TOKENIZER_PATH, expected_sha256=digest)
    statistics = {}
    roundtrips = 0
    for name, rows in (('train', training), ('development', development)):
        byte_count = token_count = 0
        for row in rows:
            text = row['text']
            ids = tokenizer.encode(text)
            assert tokenizer.decode(ids) == text
            assert restored.encode(text) == ids
            byte_count += len(text.encode('utf-8'))
            token_count += len(ids)
            roundtrips += 1
        statistics[name] = {'stories': len(rows), 'utf8_bytes': byte_count,
            'ordinary_tokens': token_count, 'bytes_per_ordinary_token': byte_count / token_count}
    probes = ['café 🐈', '  Mia\nSaw a CAT!  ', '']
    for text in probes:
        assert restored.decode(restored.encode(text, add_bos=True, add_eos=True)) == text
    metadata = {'backend': 'tokenizers', 'backend_version': version('tokenizers'),
        'requested_vocabulary': 512, 'actual_vocabulary': tokenizer.vocab_size,
        'base_byte_alphabet': 256, 'special_tokens': dict(zip(SPECIALS, [0, 1, 2])),
        'normalizer': None, 'pre_tokenizer': 'ByteLevel(add_prefix_space=False,use_regex=True)',
        'decoder': 'ByteLevel', 'min_frequency': 2, 'artifact': 'artifacts/phase02/tokenizer.json',
        'artifact_sha256': digest, 'train_file_sha256': sha256((ROOT / 'data/processed/phase02/train.jsonl').read_bytes()),
        'fitting_population': 'training stories only; development used for post-fit inspection',
        'final_holdout_access': 'not loaded, encoded or scored by this stage',
        'literal_control_strings': 'rejected in ordinary text; controls inserted by explicit ID',
        'statistics': statistics, 'roundtrip_and_reload_story_checks': roundtrips,
        'additional_roundtrip_probes': len(probes)}
    write_json(MANIFESTS / 'tokenizer.json', metadata)
    record('04_tokenizer', metadata | {
        'command': '.venv/Scripts/python scripts/phase02/04_train_tokenizer.py',
        'teaching_probe': {'text': 'Mia saw a cat.', 'ids': tokenizer.encode('Mia saw a cat.'),
                          'pieces': backend.encode('Mia saw a cat.').tokens}})
