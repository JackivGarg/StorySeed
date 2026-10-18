"""Small actual observations for the decoder and reliable-training phases."""
import json
from storyseed.corpus import ROOT,write_json


def record(phase,name,result):
    folder=ROOT/f'reports/phase{phase:02d}'
    write_json(folder/f'{name}.json',result)
    (folder/f'{name}.md').write_text(f'# Observed Phase {phase}: {name}\n\nActual execution; student understanding remains unmeasured.\n\n```json\n'+json.dumps(result,indent=2,ensure_ascii=True)+'\n```\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))
