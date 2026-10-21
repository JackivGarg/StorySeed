"""An expanded, training-only tokenizer and new reserved recovery populations."""
import json
from pathlib import Path
import urllib.request
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from storyseed.corpus import ROOT, REVISION, DATASET, SOURCE_FILE, DELIMITER, sha256, read_jsonl, write_jsonl
from storyseed.experiment_data import write_json
from storyseed.story_splits import duplicate_key, split_name
from storyseed.story_tokenizer import StoryTokenizer, SPECIALS
from storyseed.training_data import TrainingBatches

FOLDER = ROOT / 'data/processed/quality_recovery'
MANIFEST = ROOT / 'data/manifests/quality_recovery/corpus.json'
TOKENIZER = ROOT / 'artifacts/quality_recovery/tokenizer.json'
PROTOCOL = ROOT / 'planning/QUALITY_RECOVERY_PROTOCOL.json'


def plan():
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def encode(rows, name, tokenizer):
    ids, offsets = [], [0]
    for row in rows:
        sequence = tokenizer.encode(row['text'], add_bos=True, add_eos=True)
        ids.extend(sequence)
        offsets.append(len(ids))
    FOLDER.mkdir(parents=True, exist_ok=True)
    tokens, boundaries = FOLDER / f'{name}_tokens.npy', FOLDER / f'{name}_offsets.npy'
    np.save(tokens, np.asarray(ids, dtype=np.uint16))
    np.save(boundaries, np.asarray(offsets, dtype=np.int64))
    return {'stories': len(rows), 'targets': len(ids) - len(rows),
            'utf8_bytes': sum(len(row['text'].encode('utf-8')) for row in rows),
            'tokens': tokens.relative_to(ROOT).as_posix(), 'offsets': boundaries.relative_to(ROOT).as_posix(),
            'tokens_sha256': sha256(tokens.read_bytes()), 'offsets_sha256': sha256(boundaries.read_bytes())}


def prepare():
    spec = plan()
    protocol_hash = sha256(PROTOCOL.read_bytes())
    if MANIFEST.exists():
        saved = json.loads(MANIFEST.read_text(encoding='utf-8'))
        if saved['protocol_sha256'] != protocol_hash:
            raise ValueError('Existing recovery corpus belongs to another protocol.')
        return saved
    raw = ROOT / 'data/raw/quality_recovery'
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / 'source-prefix.bin'
    size = spec['source_prefix_bytes']
    if path.exists():
        prefix = path.read_bytes()
        if len(prefix) != size:
            raise ValueError('Existing recovery source prefix is incomplete.')
    else:
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SOURCE_FILE}'
        request = urllib.request.Request(url, headers={'Range': f'bytes=0-{size - 1}', 'User-Agent': 'StorySeed-quality-recovery/1'})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status not in (200, 206) or (response.status == 206 and not (response.headers.get('Content-Range') or '').startswith('bytes 0-')):
                raise ValueError('Source response is not the requested pinned prefix.')
            prefix = response.read(size)
        if len(prefix) != size:
            raise ValueError('Recovery prefix shorter than the declared budget.')
        path.write_bytes(prefix)
    old = json.loads((ROOT / 'data/manifests/phase07/corpus.json').read_text(encoding='utf-8'))
    if sha256(prefix[:spec['old_source_prefix_bytes']]) != old['prefix_sha256']:
        raise ValueError('Recovery source disagrees with the pinned original 8 MiB.')
    old_groups = {duplicate_key(segment.decode('utf-8-sig').replace('\r\n', '\n').strip())
                  for segment in prefix[:spec['old_source_prefix_bytes']].split(DELIMITER)[:-1] if segment.strip()}
    groups = {}
    for index, segment in enumerate(prefix.split(DELIMITER)[:-1]):
        text = segment.decode('utf-8-sig').replace('\r\n', '\n').strip()
        if text:
            key = duplicate_key(text)
            groups.setdefault(key, {'id': key, 'source_index': index, 'text': text})
    training = [row for key, row in sorted(groups.items()) if split_name(key) == 'train']
    development = [row for key, row in sorted(groups.items()) if key not in old_groups and split_name(key) == 'development'][:200]
    final = [row for key, row in sorted(groups.items()) if key not in old_groups and split_name(key) == 'final_holdout'][:200]
    if len(development) != 200 or len(final) != 200:
        raise ValueError('Not enough newly reserved stories.')
    populations = [set(row['id'] for row in rows) for rows in [training, development, final]]
    if populations[0] & populations[1] or populations[0] & populations[2] or populations[1] & populations[2]:
        raise ValueError('Recovery group overlap.')
    for name, rows in [('train', training), ('development', development), ('fresh_final', final)]:
        write_jsonl(FOLDER / f'{name}.jsonl', rows)
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=4096, min_frequency=2, show_progress=False,
                                   special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    backend.train_from_iterator((row['text'] for row in training), trainer=trainer, length=len(training))
    TOKENIZER.parent.mkdir(parents=True, exist_ok=True)
    backend.save(str(TOKENIZER))
    tokenizer = StoryTokenizer(backend)
    if tokenizer.vocab_size != 4096:
        raise ValueError('Unexpected fitted recovery vocabulary size.')
    manifest = {'format': 'storyseed.quality-corpus.v1', 'protocol_sha256': protocol_hash,
                'dataset': DATASET, 'revision': REVISION, 'prefix_bytes': len(prefix), 'prefix_sha256': sha256(prefix),
                'complete_normalized_unique_stories': len(groups), 'old_prefix_groups_excluded_from_new_reserved': len(old_groups),
                'group_overlap': 0, 'near_duplicates': 'normalized exact grouping only; semantic overlap is not ruled out',
                'tokenizer': TOKENIZER.relative_to(ROOT).as_posix(), 'tokenizer_sha256': sha256(TOKENIZER.read_bytes()),
                'vocabulary': tokenizer.vocab_size, 'tokenizer_fit_population': 'expanded training only',
                'files': {'train': encode(training, 'train', tokenizer), 'development': encode(development, 'development', tokenizer)},
                'fresh_final': {'stories': len(final), 'jsonl_sha256': sha256((FOLDER / 'fresh_final.jsonl').read_bytes()),
                                'encoded_or_scored': False},
                'license_declared_by_pinned_source': 'cdla-sharing-1.0', 'attribution': 'Ronen Eldan and Yuanzhi Li, TinyStories (2023)',
                'sampling': 'bounded source prefix convenience population; not representative full benchmark'}
    write_json(MANIFEST, manifest)
    return manifest


class RecoveryBatches(TrainingBatches):
    def __init__(self, name, context=256, score_tokens=128):
        metadata = json.loads(MANIFEST.read_text(encoding='utf-8'))
        item = metadata['files'][name]
        for field, attribute in [('tokens', 'ids'), ('offsets', 'offsets')]:
            path = ROOT / item[field]
            if sha256(path.read_bytes()) != item[field + '_sha256']:
                raise ValueError('Recovery numerical data changed.')
            setattr(self, attribute, np.load(path, mmap_mode='r'))
        self.context, self.score_tokens = context, score_tokens
        self.locations = [(doc, start) for doc in range(len(self.offsets) - 1)
                          for start in range(0, int(self.offsets[doc + 1] - self.offsets[doc]) - 1, score_tokens)]
        self.target_counts = [min(score_tokens, int(self.offsets[doc + 1] - self.offsets[doc]) - 1 - start) for doc, start in self.locations]
        if sum(self.target_counts) != item['targets']:
            raise ValueError('Recovery target accounting mismatch.')


def identity():
    return {'protocol_sha256': sha256(PROTOCOL.read_bytes()), 'corpus_manifest_sha256': sha256(MANIFEST.read_bytes()),
            'tokenizer_sha256': sha256(TOKENIZER.read_bytes()), 'context': 256, 'score_tokens': 128}
