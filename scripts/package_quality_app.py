"""Package a reviewed recovery release; keep the original release intact."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import zipfile

from storyseed.corpus import ROOT
from package_local_app import RUNTIME, LAUNCHER, digest, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'exports/StorySeed-quality-recovery')
    args = parser.parse_args()
    folder = args.destination.resolve()
    archive = folder.with_suffix('.zip')
    if folder.exists() or archive.exists():
        parser.error('Choose a new destination; existing releases are preserved.')
    release = json.loads((ROOT / 'artifacts/application_release.json').read_text(encoding='utf-8'))
    checkpoint, tokenizer = ROOT / release['checkpoint'], ROOT / release['tokenizer']
    if digest(checkpoint) != release['model_sha256'] or digest(tokenizer) != release['tokenizer_sha256']:
        parser.error('Application artifacts differ from the reviewed release.')
    folder.mkdir(parents=True)
    shutil.copyfile(checkpoint, folder / 'model.pt')
    shutil.copyfile(tokenizer, folder / 'tokenizer.json')
    for name in RUNTIME:
        target = folder / 'runtime/storyseed' / f'{name}.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'src/storyseed' / f'{name}.py', target)
    for relative in ['requirements-inference.txt', 'docs/LOCAL_APP.md', 'docs/RECOVERY_MODEL_CARD.md',
                     'docs/QUALITY_RECOVERY_REPORT.md', 'reports/quality_recovery/01_diagnosis.json',
                     'reports/quality_recovery/02_data.json', 'reports/quality_recovery/02_representation.json',
                     'reports/quality_recovery/04_metrics.json', 'reports/quality_recovery/04_samples.json',
                     'reports/quality_recovery/04_SAMPLE_PACKET.md', 'reports/quality_recovery/ASSISTANT_REVIEW.md']:
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    write(folder / 'launch.py', LAUNCHER)
    write(folder / 'README.md', '# StorySeed quality recovery\n\nThis is the reviewed scratch-trained recovery model, not the historical 512-entry prototype.\n\nInstall compatible Python/dependencies, then run `python launch.py --gui` or `python launch.py --json`.\nSee docs/LOCAL_APP.md and docs/RECOVERY_MODEL_CARD.md.\n\nThe bundle contains weights, tokenizer, runtime source and measured evidence, not Python/dependency binaries. Inference is local after setup. Generated stories can still contain errors. Human review remains pending.\n\nUse `python -m venv .venv`, then `.venv/Scripts/python -m pip install -r requirements-inference.txt` on Windows before running `.venv/Scripts/python launch.py --gui`.')
    files = {path.relative_to(folder).as_posix(): digest(path) for path in sorted(folder.rglob('*')) if path.is_file()}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    manifest = {'format': 'storyseed.local-release.v1', 'model': release['model'], 'context': release['context'],
                'model_sha256': release['model_sha256'], 'tokenizer_sha256': release['tokenizer_sha256'],
                'source_base_revision': revision, 'source_working_tree_dirty': dirty,
                'python_used_to_package': platform.python_version(), 'files': files}
    write(folder / 'manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                stream.write(path, str(Path(folder.name) / path.relative_to(folder)))
    report = {'folder': str(folder), 'archive': str(archive), 'archive_bytes': archive.stat().st_size,
              'archive_sha256': digest(archive), 'manifest_sha256': digest(folder / 'manifest.json'),
              'release': release, 'source_base_revision': revision, 'source_working_tree_dirty': dirty,
              'python_or_dependencies_included': False}
    write(ROOT / 'reports/quality_recovery/FINAL_RELEASE.json', json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
