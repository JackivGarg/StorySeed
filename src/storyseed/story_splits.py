"""Group normalized duplicate stories before a stable, content-based split."""
from collections import Counter
import json
import unicodedata

from storyseed.corpus import RAW, PROCESSED, MANIFESTS, read_jsonl, write_jsonl, write_json, sha256, record

SEED = 17


def duplicate_key(text: str) -> str:
    normalized = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    return sha256(normalized.encode('utf-8'))


def split_name(key: str) -> str:
    bucket = int(sha256(f'{SEED}:{key}'.encode('utf-8'))[:16], 16) % 1000
    return 'train' if bucket < 800 else ('development' if bucket < 900 else 'final_holdout')


def load_split(name: str) -> list[dict]:
    if name not in {'train', 'development', 'final_holdout'}:
        raise ValueError('Unknown split name.')
    path = PROCESSED / f'{name}.jsonl'
    manifest = json.loads((MANIFESTS / 'splits.json').read_text(encoding='utf-8'))
    if sha256(path.read_bytes()) != manifest['files'][name]['sha256']:
        raise ValueError('Split file differs from its manifest.')
    return read_jsonl(path)


def prepare_splits() -> None:
    provenance = json.loads((MANIFESTS / 'source.json').read_text(encoding='utf-8'))
    path = RAW / 'stories.jsonl'
    if sha256(path.read_bytes()) != provenance['raw_jsonl_sha256']:
        raise ValueError('Captured stories changed since source manifest creation.')
    groups = {}
    duplicates = 0
    for row in read_jsonl(path):
        key = duplicate_key(row['text'])
        if key in groups:
            duplicates += 1
            continue
        groups[key] = {'id': key, 'source_index': row['source_index'],
                       'content_sha256': row['sha256'], 'text': row['text']}
    splits = {name: [] for name in ('train', 'development', 'final_holdout')}
    assignments = {}
    for key, row in sorted(groups.items()):
        name = split_name(key)
        splits[name].append(row)
        assignments[key] = name
    if any(not rows for rows in splits.values()):
        raise ValueError('The pilot needs nonempty train, development and holdout splits.')
    keys = {name: {row['id'] for row in rows} for name, rows in splits.items()}
    assert not (keys['train'] & keys['development'] or keys['train'] & keys['final_holdout'] or keys['development'] & keys['final_holdout'])
    files = {}
    for name, rows in splits.items():
        destination = PROCESSED / f'{name}.jsonl'
        write_jsonl(destination, rows)
        files[name] = {'stories': len(rows), 'sha256': sha256(destination.read_bytes())}
    manifest = {
        'seed': SEED, 'source_manifest_sha256': sha256((MANIFESTS / 'source.json').read_bytes()),
        'duplicate_key': 'SHA-256 of NFKC + casefold + collapsed whitespace; retained text unchanged',
        'assignment': 'first 64 hash bits of 17:group-id modulo 1000; <800 train, <900 development, else holdout',
        'duplicate_records_removed': duplicates, 'unique_stories': len(groups),
        'overlap_groups': 0, 'files': files, 'assignments': assignments,
        'holdout_policy': 'reserve final_holdout; no tokenizer fitting, encoding pilots, tuning or quality scoring on it in Phase 2',
        'near_duplicate_limit': 'normalization-equivalent groups only; no semantic/paraphrase duplicate claim',
    }
    write_json(MANIFESTS / 'splits.json', manifest)
    record('02_splits', {key: value for key, value in manifest.items() if key != 'assignments'} | {
        'command': '.venv/Scripts/python scripts/phase02/02_split_stories.py',
        'normalization_fixture_same_key': duplicate_key('Mia  saw a cat.') == duplicate_key('mia saw a cat.'),
    })
