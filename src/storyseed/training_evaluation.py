"""Evaluation sums actual target losses and restores the caller's mode."""
import math
import torch
from storyseed.training_data import token_losses


@torch.no_grad()
def evaluate_decoder(model,data,batch_rows=16,max_batches=None):
    if batch_rows<1 or (max_batches is not None and max_batches<1):raise ValueError('Positive evaluation bounds required.')
    previous_mode=model.training;model.eval()
    total=0.0;targets=0;batches=0
    device=next(model.parameters()).device
    try:
        for start in range(0,len(data),batch_rows):
            if max_batches is not None and batches>=max_batches:break
            batch=data.batch(list(range(start,min(start+batch_rows,len(data)))))
            x,y,mask=[batch[key].to(device) for key in ['inputs','targets','loss_mask']]
            values=token_losses(model(x),y,mask)
            total+=values.double().sum().item();targets+=values.numel();batches+=1
    finally:model.train(previous_mode)
    if targets==0 or not math.isfinite(total):raise RuntimeError('Invalid evaluation population/loss.')
    nll=total/targets
    return {'nats_per_target_token':nll,'perplexity':math.exp(nll),'prediction_positions':targets,
            'batches':batches,'complete_population':batches*batch_rows>=len(data),'precision':'fp32 logits/loss, float64 sum'}
