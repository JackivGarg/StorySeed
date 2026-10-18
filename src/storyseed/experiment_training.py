"""Fixed-budget scratch runs with periodic measured evidence and full checkpoints."""
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from storyseed.corpus import ROOT,sha256
from storyseed.experiment_data import write_json
from storyseed.training_config import TrainingConfig
from storyseed.training_runner import TrainingRunner
from storyseed.training_checkpoint import save_training,load_training,runtime_identity
from storyseed.training_evaluation import evaluate_decoder
from storyseed.decoder import parameter_ledger
from storyseed.decoder_io import save_inference
from storyseed.experiment_data import ExperimentBatches,experiment_identity,protocol


def state_digest(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode('utf-8'));digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def generate(model,tokenizer,prompt,*,context,seed=17,max_new_tokens=128,temperature=.8,top_k=40):
    previous=model.training;model.eval();device=next(model.parameters()).device
    history=tokenizer.encode(prompt,add_bos=True);tokens=[];rng=torch.Generator().manual_seed(seed)
    started=time.perf_counter()
    try:
        for _ in range(max_new_tokens):
            values=history[-context:]
            inputs=torch.tensor([[0]*(context-len(values))+values],dtype=torch.long,device=device)
            logits=model.forward_contexts(inputs)[0].float().cpu()/temperature
            logits[:2]=float('-inf')
            if top_k is not None:
                top,indices=logits.topk(top_k);logits=torch.full_like(logits,float('-inf')).scatter(0,indices,top)
            token=torch.multinomial(logits.softmax(-1),1,generator=rng).item();tokens.append(token)
            if token==tokenizer.eos_id:break
            history.append(token)
    finally:model.train(previous)
    return {'prompt':prompt,'text':prompt+tokenizer.decode(tokens),'continuation':tokenizer.decode(tokens),
        'generated_ids':tokens,'stop_reason':'eos' if tokens and tokens[-1]==tokenizer.eos_id else 'length',
        'seed':seed,'input_context':context,'temperature':temperature,'top_k':top_k,'max_new_tokens':max_new_tokens,
        'generation_seconds':time.perf_counter()-started}


def train_condition(condition):
    spec=protocol();name=condition['id'];context=condition['input_context']
    config=TrainingConfig.load(ROOT/spec['common_training_config'])
    data=ExperimentBatches('train',context,config.score_tokens)
    development=ExperimentBatches('development',context,config.score_tokens)
    identity=experiment_identity(context)
    folder=ROOT/f'runs/phase07/{name}';folder.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/f'checkpoints/phase07/{name}-latest.pt'
    result_path=ROOT/f'reports/phase07/{name}.json'
    if result_path.exists():
        completed=json.loads(result_path.read_text(encoding='utf-8'))
        if completed['identity']!=identity or completed['successful_updates']!=config.total_updates:raise ValueError('Existing condition result is incompatible.')
        for artifact in [completed['inference'],completed['checkpoint']]:
            if sha256(Path(artifact['path']).read_bytes())!=artifact['sha256']:raise ValueError('Completed artifact changed.')
        return completed
    history_path=folder/'history.json'
    if checkpoint.exists():
        runner=load_training(checkpoint,config,data=data,identity=identity)
        history=json.loads(history_path.read_text(encoding='utf-8'))
        history['updates']=[x for x in history['updates'] if x['update']<=runner.updates]
        history['evaluations']=[x for x in history['evaluations'] if x['completed_updates']<=runner.updates]
        prior_seconds=history.get('elapsed_seconds',0.0)
    else:
        runner=TrainingRunner(config,data=data,identity=identity)
        history={'condition':condition,'identity':identity,'initial_state_sha256':state_digest(runner.model),'updates':[],'evaluations':[]}
        prior_seconds=0.0
    from storyseed.story_tokenizer import StoryTokenizer
    tokenizer=StoryTokenizer.load()
    started=time.monotonic();compute_seconds=0.0;compute_targets=0;input_count=0
    # RNG-free evaluation and per-sample local generators do not consume dropout/data RNG.
    if not history['evaluations']:
        metrics=evaluate_decoder(runner.model,development,config.evaluation_batch_rows)
        history['evaluations'].append({'completed_updates':0,'development':metrics,
            'sample':generate(runner.model,tokenizer,spec['sample_prompts'][0],context=context,max_new_tokens=96)})
    torch.cuda.reset_peak_memory_stats()

    def persist():
        history['elapsed_seconds']=prior_seconds+time.monotonic()-started
        # Log first; if interrupted before replacing the checkpoint, trim the newer log on resume.
        temporary=history_path.with_suffix('.json.tmp');write_json(temporary,history);temporary.replace(history_path)
        artifact=save_training(runner,checkpoint)
        write_json(folder/'checkpoint.json',artifact)
        return artifact

    persist()
    while runner.updates<config.total_updates:
        if prior_seconds+time.monotonic()-started>spec['maximum_seconds_per_condition']:
            persist();raise RuntimeError('Condition reached its time bound; valid checkpoint saved, completion not claimed.')
        torch.cuda.synchronize();tick=time.perf_counter()
        observation=runner.step()
        torch.cuda.synchronize();duration=time.perf_counter()-tick
        if observation['optimizer_step_skipped']:raise RuntimeError('Controlled study requires equal successful row draws; skipped update needs protocol review.')
        observation['step_seconds']=duration
        history['updates'].append(observation)
        compute_seconds+=duration;compute_targets+=observation['true_targets'];input_count+=observation['input_tokens']
        if runner.updates%50==0:print(json.dumps({'condition':name,'update':runner.updates,'nll':observation['nll'],'target_presentations':runner.optimizer_target_tokens}),flush=True)
        if runner.updates in spec['evaluation_updates']:
            metrics=evaluate_decoder(runner.model,development,config.evaluation_batch_rows)
            item={'completed_updates':runner.updates,'development':metrics}
            if runner.updates in [500,2000]:item['sample']=generate(runner.model,tokenizer,spec['sample_prompts'][0],context=context,max_new_tokens=96)
            history['evaluations'].append(item)
        if runner.updates%spec['checkpoint_every']==0:persist()
    artifact=persist()
    inference=save_inference(runner.model,ROOT/f'checkpoints/phase07/{name}-inference.pt',f'expanded-corpus scratch run; {context} supplied input slots; fixed last update')
    sampled_rows=hashlib.sha256(json.dumps([point['rows'] for point in history['updates']],separators=(',',':')).encode('utf-8')).hexdigest()
    # Derive full-run resource counts from the persisted observations, including resumed portions.
    total_seconds=sum(point['step_seconds'] for point in history['updates'])
    targets=sum(point['true_targets'] for point in history['updates'])
    inputs=sum(point['input_tokens'] for point in history['updates'])
    outcome={'condition':condition,'configuration':config.to_dict(),'identity':identity,'parameter_ledger':parameter_ledger(runner.model),
        'initial_state_sha256':history['initial_state_sha256'],'sampled_row_sequence_sha256':sampled_rows,
        'successful_updates':runner.updates,'attempts':runner.attempts,'scored_target_presentations':targets,
        'training_population_targets':sum(data.target_counts),'target_presentations_per_population':targets/sum(data.target_counts),
        'processed_real_input_tokens':inputs,'measured_update_compute_seconds':total_seconds,
        'scored_target_tokens_per_update_second':targets/total_seconds,'elapsed_seconds':history['elapsed_seconds'],
        'peak_allocated_bytes_this_process':torch.cuda.max_memory_allocated(),'peak_reserved_bytes_this_process':torch.cuda.max_memory_reserved(),
        'runtime':runtime_identity(runner),'development_curve':history['evaluations'],'first_sampled_training_nll':history['updates'][0]['nll'],
        'last_sampled_training_nll':history['updates'][-1]['nll'],'checkpoint':artifact,'inference':inference,
        'history_path':history_path.relative_to(ROOT).as_posix(),'history_sha256':sha256(history_path.read_bytes()),
        'scope':'one paired training seed; fixed expanded source prefix; last checkpoint, not optimized final-set selection',
        'final_holdout_accessed':False}
    write_json(result_path,outcome)
    # Small public observations support real curves without uploading all row logs.
    write_json(ROOT/f'reports/phase07/{name}-curve.json',{'condition':name,'history_sha256':outcome['history_sha256'],
        'updates':[{key:point[key] for key in ['update','nll','learning_rate','true_targets','gradient_norm_before_clip','step_seconds']} for point in history['updates']],
        'development':history['evaluations']})
    del runner
    torch.cuda.empty_cache()
    return outcome
