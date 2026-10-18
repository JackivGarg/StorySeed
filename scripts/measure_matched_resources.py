"""Separate short synchronized timing observation, retaining all samples and stalls."""
import gc,json,time,subprocess
import numpy as np
import torch
from storyseed.corpus import ROOT,sha256
from storyseed.experiment_data import ExperimentBatches,experiment_identity,protocol,write_json
from storyseed.training_config import TrainingConfig
from storyseed.training_runner import TrainingRunner


def main():
    spec=protocol();config=TrainingConfig.load(ROOT/spec['common_training_config'])
    try:
        observed=subprocess.run(['nvidia-smi','--query-gpu=name,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True).stdout.strip()
    except (OSError,subprocess.SubprocessError):observed='device clock/power observation unavailable'
    results=[]
    for condition in spec['conditions']:
        context=condition['input_context'];data=ExperimentBatches('train',context,config.score_tokens)
        rows=list(range(config.micro_batch_rows*config.accumulation_steps))
        for trial in range(3):
            runner=TrainingRunner(config,data=data,identity=experiment_identity(context))
            for _ in range(3):runner.step(rows)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            samples=[]
            for _ in range(10):
                torch.cuda.synchronize();started=time.perf_counter();value=runner.step(rows);torch.cuda.synchronize()
                samples.append({'seconds':time.perf_counter()-started,'true_targets':value['true_targets'],'real_inputs':value['input_tokens']})
            contaminated=any(item['seconds']>5 for item in samples)
            results.append({'condition':condition['id'],'trial':trial,'warmup_updates':3,'timed_updates':10,'fixed_rows':rows,
                'samples':samples,'contaminated_by_predeclared_over_5_second_update':contaminated,
                'scored_targets_per_second':sum(item['true_targets'] for item in samples)/sum(item['seconds'] for item in samples),
                'median_update_seconds':float(np.median([item['seconds'] for item in samples])),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            del runner;gc.collect();torch.cuda.empty_cache()
    output={'scope':'separate resource-only scratch copies; no change to frozen scientific training artifacts or final scores',
        'reason':'one original context64 update contained a 17,768-second stall; raw long-run timing remains preserved',
        'protocol_sha256':sha256((ROOT/'planning/PHASE07_08_PROTOCOL.json').read_bytes()),'trials':results,'device_observation_before_trials':observed,
        'measurement':'synchronized wall duration per full update including CPU batching; not pure kernel time',
        'comparison_valid':not any(item['contaminated_by_predeclared_over_5_second_update'] for item in results),
        'limits':'short matched observations, one machine; no long-run thermal guarantee'}
    write_json(ROOT/'reports/phase08/matched_resources.json',output)
    print(json.dumps({'matched_resource_observation_complete':True,'comparison_valid':output['comparison_valid']}),flush=True)


if __name__=='__main__':main()
