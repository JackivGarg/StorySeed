"""Reproduce saved attention inference in a fresh process without fitting."""
import json
import subprocess
import sys
from storyseed.corpus import ROOT, sha256
from storyseed.phase04_evidence import record


def main():
    checkpoint = ROOT/'checkpoints/phase04/attention.pt'
    before = sha256(checkpoint.read_bytes())
    manifest = json.loads((ROOT/'data/manifests/phase04/attention.json').read_text(encoding='utf-8'))
    assert before == manifest['sha256']
    completed = subprocess.run([sys.executable,str(ROOT/'scripts/generate_phase04.py'),'--json'],
             cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=45,check=True)
    output = json.loads(completed.stdout)
    expected = json.loads((ROOT/'reports/phase04/06_integration.json').read_text(encoding='utf-8'))[
        'samples_common_prompt_seed_temperature_top_k_budget']['phase04_attention']
    assert output == expected
    assert sha256(checkpoint.read_bytes()) == before
    record('07_cli_reload',{
        'command':'.venv/Scripts/python scripts/phase04/07_reproduce_cli.py',
        'child_command':'.venv/Scripts/python scripts/generate_phase04.py --json',
        'fresh_process_sample_matches':True,'checkpoint_sha256_unchanged':before,
        'output':output,'weights_retrained':False,'final_holdout_accessed':False})


if __name__ == '__main__':
    main()
