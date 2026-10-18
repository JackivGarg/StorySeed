"""Explicit vectorized multi-head causal retrieval and decoder sublayers."""
import math
import torch
from torch import nn


class MultiHeadCausalAttention(nn.Module):
    def __init__(self,dimensions,heads):
        super().__init__()
        if type(heads) is not int or heads<1 or dimensions<1 or dimensions%heads:
            raise ValueError('Feature width must divide into a positive integer head count.')
        self.dimensions,self.heads,self.head_dimensions=dimensions,heads,dimensions//heads
        self.query=nn.Linear(dimensions,dimensions,bias=False)
        self.key=nn.Linear(dimensions,dimensions,bias=False)
        self.value=nn.Linear(dimensions,dimensions,bias=False)
        self.output=nn.Linear(dimensions,dimensions,bias=False)

    def split_heads(self,states):
        b,t,_=states.shape
        return states.reshape(b,t,self.heads,self.head_dimensions).transpose(1,2)

    def forward(self,states,valid,*,return_weights=False):
        b,t,d=states.shape
        if d!=self.dimensions or valid.shape!=(b,t) or valid.dtype!=torch.bool:
            raise ValueError('Expected features [B,T,D] and boolean real-token flags [B,T].')
        q,k,v=[self.split_heads(layer(states)) for layer in (self.query,self.key,self.value)]
        scores=(q @ k.transpose(-2,-1))/math.sqrt(self.head_dimensions)
        causal=torch.ones(t,t,device=states.device,dtype=torch.bool).tril()
        allowed=causal[None,None] & valid[:,None,None,:] & valid[:,None,:,None]
        has_source=allowed.any(-1,keepdim=True)
        masked=scores.masked_fill(~allowed,float('-inf'))
        safe=torch.where(has_source,masked,torch.zeros_like(masked))
        weights=torch.where(has_source,safe.softmax(-1),torch.zeros_like(safe))
        mixed=weights @ v
        joined=mixed.transpose(1,2).contiguous().reshape(b,t,d)
        output=self.output(joined)
        return (output,weights) if return_weights else output
