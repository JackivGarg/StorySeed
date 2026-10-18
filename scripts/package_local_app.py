"""Create a new local source/weight release without copying corpora or environments."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import zipfile

from storyseed.app_service import MODEL_SHA256, TOKENIZER_SHA256
from storyseed.corpus import ROOT

RUNTIME = ['__init__', 'app', 'app_service', 'desktop', 'release_bundle', 'context_generation',
           'decoder_io', 'decoder', 'decoder_components', 'position_features', 'token_features',
           'bigram', 'story_tokenizer', 'corpus', 'story_splits']

LAUNCHER = '''"""Run the bundled source with an already installed compatible Python runtime."""
import hashlib, json, pathlib, sys
folder = pathlib.Path(__file__).resolve().parent
manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
for relative, digest in manifest['files'].items():
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit('Missing or changed bundle file: ' + relative)
sys.path.insert(0, str(folder / 'runtime'))
from storyseed.app import main
sys.argv[1:1] = ['--bundle', str(folder)]
main()
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + '\n', encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'exports/StorySeed-main128')
    args = parser.parse_args()
    folder = args.destination.resolve()
    archive = folder.with_suffix('.zip')
    if folder.exists() or archive.exists():
        parser.error('Destination or sibling ZIP exists; choose a new destination to preserve the prior release.')
    checkpoint = ROOT / 'checkpoints/phase07/main128-inference.pt'
    tokenizer = ROOT / 'artifacts/phase02/tokenizer.json'
    if digest(checkpoint) != MODEL_SHA256 or digest(tokenizer) != TOKENIZER_SHA256:
        parser.error('Source artifacts do not match the fixed main128 release.')
    folder.mkdir(parents=True)
    shutil.copyfile(checkpoint, folder / 'model.pt')
    shutil.copyfile(tokenizer, folder / 'tokenizer.json')
    for module in RUNTIME:
        path = folder / 'runtime/storyseed' / f'{module}.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'src/storyseed' / f'{module}.py', path)
    for name in ['requirements-inference.txt', 'docs/LOCAL_APP.md', 'docs/MODEL_CARD.md', 'docs/UNIVERSITY_REPORT.md', 'docs/DEMO_GUIDE.md',
                 'docs/RECONSTRUCTION_WORKBOOK.md', 'reports/phase08/PAPER_DRAFT.md']:
        source = ROOT / name
        if source.is_file():
            target = folder / source.name
            shutil.copyfile(source, target)
    provenance = folder / 'provenance'
    provenance.mkdir()
    for name, relative in {'corpus.json': 'data/manifests/phase07/corpus.json',
                           'protocol.json': 'planning/PHASE07_08_PROTOCOL.json',
                           'inference.json': 'reports/phase07/04_fresh_inference.json'}.items():
        shutil.copyfile(ROOT / relative, provenance / name)
    write(folder / 'launch.py', LAUNCHER)
    write(folder / 'README.md', '# StorySeed local release\n\nInstall Python 3.12 with Tcl/Tk, create a virtual environment, and install requirements-inference.txt.\nRun `python launch.py --gui` or `python launch.py --json`. See LOCAL_APP.md.\n\nThis folder contains runtime source, trained tensors, tokenizer, provenance and documentation. It does not include Python, third-party packages or raw stories. After dependencies are installed, inference is local and needs no network.\n\nThis learning prototype can produce malformed text and lose continuity. Packaging changes usability, not learned quality. manifest.json records file integrity; it is not a cryptographic publisher signature.')
    files = {str(p.relative_to(folder)).replace('\\', '/'): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT, text=True).strip())
    manifest = {'format': 'storyseed.local-release.v1', 'model': 'StorySeed-main128', 'context': 128,
                'model_sha256': MODEL_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256,
                'source_base_revision': revision, 'source_working_tree_dirty': dirty,
                'source_identity': 'individual runtime file hashes below; base revision alone may precede staged source',
                'python_used_to_package': platform.python_version(), 'files': files}
    write(folder / 'manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                stream.write(path, str(Path(folder.name) / path.relative_to(folder)))
    report = {'folder': str(folder), 'archive': str(archive), 'archive_bytes': archive.stat().st_size,
              'archive_sha256': digest(archive), 'manifest_sha256': digest(folder / 'manifest.json'),
              'files': len(files), 'model_sha256': MODEL_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256,
              'source_base_revision': revision, 'source_working_tree_dirty': dirty,
              'raw_corpus_included': False, 'python_or_dependencies_included': False}
    report_path = ROOT / 'reports/phase09' / ('03_package-' + folder.name + '.json')
    write(report_path, json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
