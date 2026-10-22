"""Load a fixed local release using paths relative to its manifest."""
import json
from pathlib import Path, PurePosixPath

from storyseed.app_service import GenerationService
from storyseed.corpus import sha256


def load_bundle(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 'storyseed.local-release.v1':
        raise ValueError('Expected a StorySeed local-release manifest.')
    if type(manifest.get('context')) is not int or manifest['context'] < 1:
        raise ValueError('The release must declare a positive input context.')
    files = manifest.get('files')
    if not isinstance(files, dict) or files.get('model.pt') != manifest.get('model_sha256') or files.get('tokenizer.json') != manifest.get('tokenizer_sha256'):
        raise ValueError('Manifest must include the exact model and tokenizer bytes.')
    for relative, expected in files.items():
        name = PurePosixPath(relative)
        if name.is_absolute() or '..' in name.parts or '\\' in relative:
            raise ValueError('Release file names must stay inside the bundle.')
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f'Missing or changed release file: {relative}')
    release = {'format': 'storyseed.application-release.v1', 'model': manifest['model'], 'context': manifest['context'],
               'model_sha256': manifest['model_sha256'], 'tokenizer_sha256': manifest['tokenizer_sha256']}
    return GenerationService(folder / 'model.pt', folder / 'tokenizer.json', release=release)
