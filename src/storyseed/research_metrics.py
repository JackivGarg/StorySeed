"""Likelihood and continuation surface measurements with explicit denominators."""
import math,re
import numpy as np
import torch
from storyseed.training_data import token_losses

@torch.no_grad()
def evaluate_by_story(model,data,batch_rows=8):
    model.eval();device=next(model.parameters()).device
    losses=np.zeros(len(data.offsets)-1,dtype=np.float64)
    counts=np.zeros(len(losses),dtype=np.int64)
    for begin in range(0,len(data),batch_rows):
        rows=list(range(begin,min(begin+batch_rows,len(data))))
        batch=data.batch(rows)
        x,y,mask=[batch[key].to(device) for key in ['inputs','targets','loss_mask']]
        logits=model(x)
        for b,row in enumerate(rows):
            values=token_losses(logits[b:b+1],y[b:b+1],mask[b:b+1])
            doc,_=data.locations[row];losses[doc]+=values.double().sum().item();counts[doc]+=values.numel()
    if not np.isfinite(losses).all() or np.any(counts<=0):raise ValueError('Invalid per-story likelihood population.')
    nll=float(losses.sum()/counts.sum())
    return {'nats_per_target_token':nll,'perplexity':math.exp(nll),'prediction_positions':int(counts.sum()),
        'stories':len(losses),'story_loss_sums':losses.tolist(),'story_target_counts':counts.tolist(),
        'precision':'FP32 model/loss; float64 sums','population':'original reserved stories; all next-ID events including EOS'}

def continuation_metrics(text):
    words=re.findall(r"\b[\w']+\b",text.casefold());grams=[tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]
    return {'continuation_words':len(words),'unique_word_fraction':len(set(words))/len(words) if words else None,
        'duplicate_word_trigram_fraction':1-len(set(grams))/len(grams) if grams else None,
        'scope':'continuation only; lowercase word tokens; lexical diversity is length-dependent, not a coherence score'}
