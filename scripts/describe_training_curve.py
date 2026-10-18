"""Describe existing curve observations; never fit or evaluate a model."""
import argparse,json,math
from pathlib import Path
from storyseed.corpus import ROOT,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--condition',default='main128',choices=['main128','context64'])
    args=parser.parse_args()
    data=json.loads((ROOT/f'reports/phase07/{args.condition}.json').read_text(encoding='utf-8'))
    curve=data['development_curve'];first=curve[0]['development']['nats_per_target_token'];last=curve[-1]['development']['nats_per_target_token']
    delta=first-last
    report={'condition':args.condition,'initial_full_development_nll':first,'final_full_development_nll':last,
        'decrease_nats_per_target':delta,'geometric_mean_correct_target_probability_factor':math.exp(delta),
        'same_development_target_count':len({point['development']['prediction_positions'] for point in curve})==1,
        'development_observations':len(curve),'sampled_training_loss_timing':'before its associated update, changing selected rows and dropout',
        'development_timing':'after the declared number of completed updates, all fixed targets, dropout off',
        'interpretation_limit':'likelihood change is not accuracy or a story-coherence score; no component-specific causal attribution'}
    write_json(ROOT/f'reports/phase07/{args.condition}-loss-interpretation.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
