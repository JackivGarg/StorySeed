"""Durable observations for the causal-attention learning journey."""
import json
from storyseed.corpus import ROOT, write_json


def record(name, result):
    folder = ROOT/'reports/phase04'
    write_json(folder/f'{name}.json', result)
    (folder/f'{name}.md').write_text(
        f'# Observed Phase 4 result: {name}\n\n'
        'Actual recorded execution; learner understanding remains unmeasured.\n\n'
        '```json\n'+json.dumps(result, indent=2, ensure_ascii=True)+'\n```\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))
