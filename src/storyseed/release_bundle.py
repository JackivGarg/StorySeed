"""Load a fixed local release using paths relative to its manifest."""
import json
from pathlib import Path, PurePosixPath

from storyseed.app_service import GenerationService, MODEL_SHA256, TOKENIZER_SHA256
from storyseed.corpus import sha256


def load_bundle(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 'storyseed.local-release.v1' or manifest.get('model') != 'StorySeed-main128':
        raise ValueError('Expected the StorySeed main128 local-release manifest.')
    if manifest.get('context') != 128 or manifest.get('model_sha256') != MODEL_SHA256 or manifest.get('tokenizer_sha256') != TOKENIZER_SHA256:
        raise ValueError('Release identity/context does not match the fixed main model.')
    files = manifest.get('files')
    if not isinstance(files, dict) or files.get('model.pt') != MODEL_SHA256 or files.get('tokenizer.json') != TOKENIZER_SHA256:
        raise ValueError('Manifest must include the exact model and tokenizer bytes.')
    for relative, expected in files.items():
        name = PurePosixPath(relative)
        if name.is_absolute() or '..' in name.parts or '\\' in relative:
            raise ValueError('Release file names must stay inside the bundle.')
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f'Missing or changed release file: {relative}')
    return GenerationService(folder / 'model.pt', folder / 'tokenizer.json')
