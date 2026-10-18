from dataclasses import replace
import time
import torch
from storyseed.training_config import TrainingConfig
from storyseed.training_runner import TrainingRunner
from storyseed.training_data import token_losses
from storyseed.build_evidence import record


def main():
    available=torch.cuda.is_available()
    modes=['fp32']+(['fp16']+(['bf16'] if torch.cuda.is_bf16_supported() else []) if available else [])
    observations={};reference=None
    for precision in modes:
        config=TrainingConfig(model=replace(TrainingConfig().model,dropout=0.0),device='cuda' if available else 'cpu',precision=precision)
        runner=TrainingRunner(config);rows=[0,1,17,len(runner.data)-1]
        batch=runner.data.batch(rows)
        x,y,mask=[batch[key].to(runner.device) for key in ['inputs','targets','loss_mask']]
        with torch.no_grad(),runner.autocast():
            logits=runner.model(x);nll=token_losses(logits,y,mask).mean().item()
        if reference is None:reference=nll
        dtype=str(logits.dtype)
        if available:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();result=runner.step(rows)
        if available:torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        observations[precision]={'forward_logits_dtype':dtype,'initial_nll':nll,'absolute_nll_difference_from_fp32':abs(nll-reference),
            'one_update_seconds':seconds,'optimizer_step_skipped':result['optimizer_step_skipped'],
            'unscaled_gradient_norm':result['gradient_norm_before_clip'],'loss_scale':result['loss_scale'],
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if available else None,
            'scope':'single small workload including first optimizer-state allocation; not a stabilized throughput benchmark'}
        del logits,x,y,mask,runner
        if available:torch.cuda.empty_cache()
    record(6,'04_precision',{'command':'.venv/Scripts/python scripts/phase06/04_inspect_precision.py',
       'torch_version':str(torch.__version__),'cuda_available':available,
       'device':torch.cuda.get_device_name(0) if available else 'cpu','observations':observations,
       'policy':'fp32 master parameters; autocast selected CUDA operations; CE in fp32; fp16 scaler unscaled once after accumulation',
       'limits':'finite one-update evidence does not establish long-run numeric stability or quality','final_holdout_accessed':False})

if __name__=='__main__':main()
