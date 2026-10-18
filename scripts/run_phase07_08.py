"""Finite training/evaluation workload for the predeclared StorySeed experiment."""
import json
import subprocess
import sys
from pathlib import Path
from storyseed.corpus import ROOT,sha256
from storyseed.experiment_data import write_json
from storyseed.experiment_data import prepare_corpus,protocol,PROTOCOL
from storyseed.experiment_training import train_condition
from storyseed.research_evaluation import evaluate_study


def main():
    # Fail on unexpected protocol edits rather than continuing a changed research plan.
    frozen=json.loads((ROOT/'planning/PHASE07_08_FREEZE.json').read_text(encoding='utf-8'))
    if frozen['protocol_sha256']!=sha256(PROTOCOL.read_bytes()):raise ValueError('Frozen protocol changed.')
    config=ROOT/protocol()['common_training_config']
    if frozen['training_config_sha256']!=sha256(config.read_bytes()):raise ValueError('Frozen training config changed.')
    state_path=ROOT/'reports/phase07_08_STATUS.json'
    if state_path.exists() and json.loads(state_path.read_text(encoding='utf-8')).get('results_complete'):
        # Reuse verified completed artifacts; preserve reviewed notes, samples and timing observations.
        for condition in protocol()['conditions']:
            if not (ROOT/f"reports/phase07/{condition['id']}.json").exists():
                raise ValueError('Completed status is missing a condition result; inspect before retrying.')
            train_condition(condition)  # Completed branch verifies data/identity/artifact hashes and performs no update.
        if not (ROOT/'reports/phase08/metrics.json').exists():raise ValueError('Completed status is missing final metrics.')
        print(json.dumps({'completed':True,'reused_existing_artifacts':True,'training_updates_performed':0,'reviewed_outputs_preserved':True}),flush=True)
        return
    write_json(state_path,{'state':'preparing expanded corpus','results_complete':False,'learner_understanding':'pending'})
    corpus=prepare_corpus()
    write_json(ROOT/'reports/phase07/01_corpus.json',corpus)
    conditions=[]
    for condition in protocol()['conditions']:
        write_json(state_path,{'state':'training '+condition['id'],'results_complete':False,'learner_understanding':'pending'})
        # Isolate GPU caches and RNG side effects between independent conditions.
        subprocess.run([sys.executable,str(ROOT/'scripts/train_experiment_condition.py'),'--condition',condition['id']],cwd=ROOT,check=True)
        conditions.append(json.loads((ROOT/f"reports/phase07/{condition['id']}.json").read_text(encoding='utf-8')))
    subprocess.run([sys.executable,str(ROOT/'scripts/measure_matched_resources.py')],cwd=ROOT,check=True)
    write_json(state_path,{'state':'frozen final evaluation and sample generation','results_complete':False,'learner_understanding':'pending'})
    evaluation=evaluate_study(conditions)
    # Fresh public inference is a separate foreground process, with no training.
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_main_export.py')],cwd=ROOT,check=True)
    for condition in protocol()['conditions']:
        subprocess.run([sys.executable,str(ROOT/'scripts/describe_training_curve.py'),'--condition',condition['id']],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/summarize_phase07_08.py')],cwd=ROOT,check=True)
    write_json(state_path,{'state':'training, frozen evaluation, plots and draft artifacts completed','results_complete':True,
        'human_review':'pending, blank ratings supplied','assistant_interpretation_and_result_commits':'pending next Codex continuation',
        'learner_understanding':'pending','main_export':conditions[0]['inference'],'final_population_scored':True})
    print(json.dumps({'completed':True,'main_export':conditions[0]['inference'],'metrics':'reports/phase08/metrics.json','review_packet':'reports/phase08/PAIRED_SAMPLE_REVIEW.md','paper_draft':'reports/phase08/PAPER_DRAFT.md','human_review':'pending'}),flush=True)


if __name__=='__main__':main()
