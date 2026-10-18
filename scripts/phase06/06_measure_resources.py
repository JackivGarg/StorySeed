from dataclasses import replace
import time
import torch
from storyseed.corpus import ROOT
from storyseed.decoder import DecoderConfig,parameter_ledger
from storyseed.training_config import TrainingConfig
from storyseed.training_runner import TrainingRunner
from storyseed.training_checkpoint import save_training,load_training,cpu_tree
from storyseed.build_evidence import record


def main():
    available=torch.cuda.is_available();precision_resume=None
    if available:
        small=TrainingRunner(replace(TrainingConfig(),device='cuda',precision='fp16'))
        small.step();small.step()
        before_overflow=cpu_tree(small.model.state_dict())
        before_update=small.updates;before_scale=small.scaler.get_scale()
        hook=small.model.output.weight.register_hook(lambda gradient:torch.full_like(gradient,float('inf')))
        skipped=small.step();hook.remove()
        unchanged=all(torch.equal(before_overflow[name],value.detach().cpu()) for name,value in small.model.state_dict().items())
        assert skipped['optimizer_step_skipped'] and small.updates==before_update and unchanged
        overflow={'controlled_gradient_overflow_skips_optimizer':True,'weights_unchanged':unchanged,
                  'successful_update_counter_unchanged':small.updates==before_update,
                  'loss_scale_before':before_scale,'loss_scale_after':small.scaler.get_scale()}
        path=ROOT/'checkpoints/phase06/fp16-resume.pt';save_training(small,path)
        saved_scale=small.scaler.get_scale();expected_next=small.step()
        expected=cpu_tree(small.model.state_dict())
        restored=load_training(path);restored_scale=restored.scaler.get_scale();actual_next=restored.step()
        maximum=max((expected[name]-value.detach().cpu()).abs().max().item() for name,value in restored.model.state_dict().items())
        assert maximum<1e-5 and saved_scale==restored_scale
        precision_resume={'saved_scale':saved_scale,'restored_scale':restored_scale,'next_rows_equal':expected_next['rows']==actual_next['rows'],
                          'next_update_max_weight_difference':maximum,'controlled_overflow':overflow,
                          'scope':'same-process CUDA fp16 state-restoration observation after a deliberately skipped attempt'}
        assert precision_resume['next_rows_equal']
        del small,restored,expected;torch.cuda.empty_cache()
    candidate=TrainingConfig.load(ROOT/'configs/phase06_candidate.json')
    if available:
        config=replace(candidate,precision='bf16' if torch.cuda.is_bf16_supported() else 'fp16')
        free_before,total_memory=torch.cuda.mem_get_info()
    else:
        config=replace(candidate,model=DecoderConfig(dimensions=64,heads=4,layers=2),score_tokens=32,
                       device='cpu',precision='fp32',micro_batch_rows=2,accumulation_steps=1)
        free_before=total_memory=None
    runner=TrainingRunner(config)
    if available:torch.cuda.reset_peak_memory_stats()
    deadline=time.monotonic()+35
    warmup=[]
    for _ in range(2):
        if time.monotonic()>deadline:raise RuntimeError('Resource-pilot deadline reached.')
        warmup.append(runner.step())
    if available:torch.cuda.synchronize()
    started=time.perf_counter();history=[]
    for _ in range(3):
        if time.monotonic()>deadline:raise RuntimeError('Resource-pilot deadline reached.')
        history.append(runner.step())
    if available:torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    targets=sum(item['true_targets'] for item in history)
    inputs=sum(item['input_tokens'] for item in history)
    successful=sum(not item['optimizer_step_skipped'] for item in history)
    assert successful==3
    seconds_per_update=elapsed/successful
    record(6,'06_resources',{'command':'.venv/Scripts/python scripts/phase06/06_measure_resources.py',
       'device':torch.cuda.get_device_name() if available else 'cpu','torch_version':str(torch.__version__),
       'measured_configuration':config.to_dict(),'parameter_ledger':parameter_ledger(runner.model),
       'warmup_attempts':len(warmup),'timed_successful_updates':successful,'timed_seconds':elapsed,
       'scored_target_tokens':targets,'processed_real_input_tokens':inputs,
       'scored_target_tokens_per_second':targets/elapsed,'processed_input_tokens_per_second':inputs/elapsed,
       'seconds_per_optimizer_update':seconds_per_update,
       'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated() if available else None,
       'peak_torch_reserved_bytes':torch.cuda.max_memory_reserved() if available else None,
       'device_free_bytes_before_candidate':free_before,'device_total_bytes':total_memory,
       'projected_2000_update_compute_seconds':2000*seconds_per_update,
       'projection_scope':'short stabilized pilot extrapolation; excludes periodic evaluation/checkpoint overhead and thermal variation',
       'cuda_fp16_resume_observation':precision_resume,
       'main_training_launched':False,'candidate_corpus_scope':'existing teaching prefix only; main corpus/protocol still to freeze in Phase 7',
       'final_holdout_accessed':False})

if __name__=='__main__':main()
