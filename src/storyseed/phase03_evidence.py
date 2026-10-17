"""Small durable observations for the learned-representation teaching phase."""
import json
from storyseed.corpus import ROOT, write_json

REPORTS = ROOT / 'reports/phase03'


def record(name, result):
    write_json(REPORTS / f'{name}.json', result)
    (REPORTS / f'{name}.md').write_text(
        f'# Observed Phase 3 result: {name}\n\n'
        'Actual command output; learner understanding remains unmeasured.\n\n'
        '```json\n' + json.dumps(result, indent=2, ensure_ascii=True) + '\n```\n',
        encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))
