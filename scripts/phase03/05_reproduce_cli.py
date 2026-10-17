"""Record fresh-process inference restoration without fitting any weights."""
import json
import subprocess
import sys
from storyseed.corpus import ROOT, sha256
from storyseed.phase03_evidence import record


def main():
    checkpoint = ROOT/'checkpoints/phase03/context_k8.pt'
    before = sha256(checkpoint.read_bytes())
    manifest = json.loads((ROOT/'data/manifests/phase03/context_k8.json').read_text(encoding='utf-8'))
    assert before == manifest['sha256']
    command = [sys.executable, str(ROOT/'scripts/generate_phase03.py'), '--json']
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               encoding='utf-8', timeout=45, check=True)
    output = json.loads(completed.stdout)
    expected = json.loads((ROOT/'reports/phase03/04_limits.json').read_text(encoding='utf-8'))[
        'samples_same_prompt_seed_temperature_top_k_budget']['context_k8']
    assert output == expected
    assert sha256(checkpoint.read_bytes()) == before
    record('05_cli_reload', {
        'command': '.venv/Scripts/python scripts/phase03/05_reproduce_cli.py',
        'child_command': '.venv/Scripts/python scripts/generate_phase03.py --json',
        'fresh_process_sample_matches': True, 'checkpoint_sha256_unchanged': before,
        'output': output, 'weights_retrained': False, 'final_holdout_accessed': False})


if __name__ == '__main__':
    main()
