"""Prepare paired continuations and explicitly pending human rating forms."""
import csv
import numpy as np
from storyseed.corpus import ROOT
from storyseed.experiment_data import write_json

def write_review_packet(samples,spec):
    packet=ROOT/'reports/phase08/PAIRED_SAMPLE_REVIEW.md'
    content='# Paired continuation review packet\n\nAssistant-generated packet; human ratings remain pending. Prompts and generation settings were frozen before training. Assess only the continuation, with the prompt visible. For each dimension use 1 poor, 2 weak, 3 mixed, 4 good, 5 strong; mark uncertainty. Keep reviewer identity and date. Model identities are in review_key.json; do not open it before a blinded review. This packet alone is not an independent human study.\n\n'
    keys=[];rng=np.random.default_rng(1987)
    for prompt_index,prompt in enumerate(spec['sample_prompts']):
        for seed in spec['sample_seeds']:
            pair=[x for x in samples if x['prompt_index']==prompt_index and x['seed']==seed]
            if rng.integers(2):pair.reverse()
            content+=f'## Pair {prompt_index+1}, sampling seed {seed}\n\nPrompt: {prompt}\n\n'
            for label,item in zip(['A','B'],pair):
                content+=f"### Continuation {label}\n\n{item['continuation']}\n\n"
                keys.append({'prompt_index':prompt_index,'seed':seed,'label':label,'model':item['model']})
    packet.write_text(content,encoding='utf-8');write_json(ROOT/'reports/phase08/review_key.json',keys)
    with (ROOT/'reports/phase08/review_ratings.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['reviewer','date','prompt_index','seed','label','grammar','prompt_continuity','local_consistency','repetition','comment'])
        for item in keys:writer.writerow(['','',item['prompt_index'],item['seed'],item['label'],'','','','',''])
