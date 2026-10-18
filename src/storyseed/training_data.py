"""Warm-up prefixes with nonoverlapping scored target blocks."""
import torch
from torch.nn import functional as F
from storyseed.story_batches import DocumentBatches


class TrainingBatches:
    def __init__(self,split,context=64,score_tokens=32):
        if type(context) is not int or type(score_tokens) is not int or not 1<=score_tokens<=context:
            raise ValueError('Require 1 <= score_tokens <= context.')
        source=DocumentBatches(split,context=score_tokens)
        self.ids,self.offsets,self.locations=source.ids,source.offsets,source.locations
        self.context,self.score_tokens=context,score_tokens
        self.target_counts=[min(score_tokens,int(self.offsets[doc+1]-self.offsets[doc])-1-start) for doc,start in self.locations]

    def __len__(self):return len(self.locations)

    def batch(self,rows):
        rows=list(rows)
        if not rows or any(type(row) is not int or not 0<=row<len(self) for row in rows):raise ValueError('Valid integer training rows required.')
        x=torch.zeros(len(rows),self.context,dtype=torch.long);y=torch.zeros_like(x)
        mask=torch.zeros_like(x,dtype=torch.bool)
        for b,row in enumerate(rows):
            document,start=self.locations[row]
            a,z=int(self.offsets[document]),int(self.offsets[document+1])
            sequence=self.ids[a:z]
            begin=max(0,start-(self.context-self.score_tokens))
            end=min(start+self.score_tokens,len(sequence)-1)
            size=end-begin
            x[b,:size]=torch.tensor(sequence[begin:end].astype('int64'))
            y[b,:size]=torch.tensor(sequence[begin+1:end+1].astype('int64'))
            mask[b,start-begin:size]=True
        return {'inputs':x,'targets':y,'loss_mask':mask,'attention_mask':x!=0,'rows':rows}


def token_losses(logits,targets,mask):
    if logits.shape[:-1]!=targets.shape or targets.shape!=mask.shape or mask.dtype!=torch.bool or not mask.any():
        raise ValueError('Expected logits [B,T,V], labels/mask [B,T], at least one real target.')
    return F.cross_entropy(logits[mask].float(),targets[mask],reduction='none')
