"""Frozen-artifact evaluation, paired story uncertainty and inspectable sample packets."""
import csv
import json
import math
import re
from pathlib import Path
import numpy as np
import torch
from storyseed.corpus import ROOT,write_json,sha256
from storyseed.experiment_data import protocol,PROTOCOL,MANIFEST,ExperimentBatches,encode_rows
from storyseed.experiment_training import generate
from storyseed.story_tokenizer import StoryTokenizer
from storyseed.story_splits import load_split
from storyseed.story_baseline import transition_counts,load_baseline
from storyseed.bigram import BigramLanguageModel
from storyseed.decoder_io import load_inference
from storyseed.research_metrics import evaluate_by_story,continuation_metrics
from storyseed.context_comparison import paired_bootstrap
from storyseed.sample_review import write_review_packet

FINAL_MANIFEST=ROOT/'data/manifests/phase08/final.json'








def release_final(conditions):
    # This transition happens only after both predeclared final artifacts exist and are fixed.
    if len(conditions)!=2:raise ValueError('Both predeclared conditions are required.')
    a,b=conditions
    for key in ['initial_state_sha256','sampled_row_sequence_sha256','scored_target_presentations','successful_updates']:
        if a[key]!=b[key]:raise ValueError('Matched training controls failed before final-set release: '+key)
    if a['parameter_ledger']['total']!=b['parameter_ledger']['total']:raise ValueError('Architecture parameter counts differ.')
    frozen={result['condition']['id']:result['inference'] for result in conditions}
    for result in conditions:
        if result['successful_updates']!=protocol()['successful_updates_per_condition']:raise ValueError('Do not release final data for a partial run.')
        artifact=result['inference']
        if sha256(Path(artifact['path']).read_bytes())!=artifact['sha256']:raise ValueError('Final model artifact changed.')
    if FINAL_MANIFEST.exists():
        prior=json.loads(FINAL_MANIFEST.read_text(encoding='utf-8'))
        if prior['frozen_artifacts']!=frozen:raise ValueError('Final population is already attached to different artifacts; no silent retuning.')
        return prior
    rows=load_split('final_holdout')
    corpus=json.loads(MANIFEST.read_text(encoding='utf-8'))
    metadata={'format':'storyseed.final-population.v1','protocol_sha256':sha256(PROTOCOL.read_bytes()),
        'tokenizer_sha256':corpus['tokenizer_sha256'],'original_split_manifest_sha256':corpus['original_split_manifest_sha256'],
        'frozen_artifacts':frozen,'release':'after both full training runs and fixed inference exports; no use in training/tokenizer/selection',
        'files':{'final':encode_rows(rows,'final',ROOT/'data/processed/phase08')},
        'source_split_sha256':sha256((ROOT/'data/processed/phase02/final_holdout.jsonl').read_bytes())}
    write_json(FINAL_MANIFEST,metadata);return metadata


def evaluate_study(conditions):
    torch.set_num_threads(1)
    spec=protocol();destination=ROOT/'reports/phase08/metrics.json'
    final=release_final(conditions)
    if destination.exists():
        previous=json.loads(destination.read_text(encoding='utf-8'))
        if previous['final_manifest_sha256']!=sha256(FINAL_MANIFEST.read_bytes()):raise ValueError('Recorded evaluation belongs to a different freeze.')
        return previous
    tokenizer=StoryTokenizer.load();results={};samples=[]
    for condition in spec['conditions']:
        name=condition['id'];context=condition['input_context']
        completed=next(result for result in conditions if result['condition']['id']==name)
        model,_=load_inference(Path(completed['inference']['path']));model=model.cuda()
        dataset=ExperimentBatches('final',context,spec['scored_block_width'],FINAL_MANIFEST)
        results[name]=evaluate_by_story(model,dataset)
        for prompt_index,prompt in enumerate(spec['sample_prompts']):
            for seed in spec['sample_seeds']:
                generated=generate(model,tokenizer,prompt,context=context,seed=seed,max_new_tokens=spec['sample_max_new_tokens'],temperature=spec['sample_temperature'],top_k=spec['sample_top_k'])
                samples.append({'model':name,'prompt_index':prompt_index,**generated,'metrics':continuation_metrics(generated['continuation'])})
        del model
        torch.cuda.empty_cache()
    training=ExperimentBatches('train',128,spec['scored_block_width'])
    counts=transition_counts(training,tokenizer.vocab_size)
    model=BigramLanguageModel(tokenizer.vocab_size)
    with torch.no_grad():model.scores.copy_((counts+.1).log())
    final_data=ExperimentBatches('final',128,spec['scored_block_width'],FINAL_MANIFEST)
    results['expanded_count_bigram_alpha_0.1']=evaluate_by_story(model,final_data)
    artifact=ROOT/'checkpoints/phase08/count-bigram.pt';artifact.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'scores':model.scores.detach().cpu(),'alpha':.1,'training_manifest_sha256':sha256(MANIFEST.read_bytes())},artifact)
    old_model,_=load_baseline()
    results['historical_phase02_learned_bigram']=evaluate_by_story(old_model,final_data)
    if any(value['story_target_counts']!=results['main128']['story_target_counts'] for value in results.values()):raise ValueError('Models did not score the identical target population.')
    paired=paired_bootstrap(results['main128'],results['context64'],spec['bootstrap_replicates'],spec['bootstrap_seed'])
    common=conditions[0];short=conditions[1]
    controls={'parameter_count_equal':common['parameter_ledger']['total']==short['parameter_ledger']['total'],
        'initial_model_state_equal':common['initial_state_sha256']==short['initial_state_sha256'],
        'sampled_training_rows_equal':common['sampled_row_sequence_sha256']==short['sampled_row_sequence_sha256'],
        'successful_target_presentations_equal':common['scored_target_presentations']==short['scored_target_presentations'],
        'successful_updates_equal':common['successful_updates']==short['successful_updates'],'same_final_target_population':True,
        'training_seeds_per_condition':1,'compute_matched':False,'caveat':'input-dependent dropout draws and window-relative positions differ with supplied context; interpreted as a context-policy intervention'}
    if not all(controls[key] for key in ['parameter_count_equal','initial_model_state_equal','sampled_training_rows_equal','successful_target_presentations_equal','successful_updates_equal']):raise ValueError('Predeclared matched controls failed; do not claim a controlled result.')
    aggregate={}
    for name in ['main128','context64']:
        selected=[item for item in samples if item['model']==name]
        aggregate[name]={'samples':len(selected),'eos_stop_fraction':sum(item['stop_reason']=='eos' for item in selected)/len(selected),
            'mean_generated_tokens':float(np.mean([len(item['generated_ids']) for item in selected])),
            'mean_generation_seconds':float(np.mean([item['generation_seconds'] for item in selected]))}
        for metric in ['unique_word_fraction','duplicate_word_trigram_fraction']:
            valid=[item['metrics'][metric] for item in selected if item['metrics'][metric] is not None]
            aggregate[name]['mean_'+metric]=float(np.mean(valid)) if valid else None
    output={'protocol_sha256':sha256(PROTOCOL.read_bytes()),'final_manifest_sha256':sha256(FINAL_MANIFEST.read_bytes()),
        'models':results,'paired_story_comparison':paired,'controls':controls,'generation_samples':len(samples),'generation_aggregate':aggregate,
        'count_bigram_artifact':{'path':str(artifact),'sha256':sha256(artifact.read_bytes())},
        'human_ratings':'pending; no human scores or agreement statistics fabricated','final_set_now_encoded_and_scored':True}
    write_json(ROOT/'reports/phase08/samples.json',samples)
    write_review_packet(samples,spec)
    write_json(destination,output);return output
