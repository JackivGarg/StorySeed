"""Explicit vectorized multi-head causal retrieval and decoder sublayers."""
import math
import torch
from torch import nn


class MultiHeadCausalAttention(nn.Module):
    def __init__(self,dimensions,heads,dropout=0.0):
        super().__init__()
        if type(heads) is not int or heads<1 or dimensions<1 or dimensions%heads:
            raise ValueError('Feature width must divide into a positive integer head count.')
        self.dimensions,self.heads,self.head_dimensions=dimensions,heads,dimensions//heads
        self.query=nn.Linear(dimensions,dimensions,bias=False)
        self.key=nn.Linear(dimensions,dimensions,bias=False)
        self.value=nn.Linear(dimensions,dimensions,bias=False)
        self.output=nn.Linear(dimensions,dimensions,bias=False)
        self.weight_dropout=nn.Dropout(dropout)

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
        mixed=self.weight_dropout(weights) @ v
        joined=mixed.transpose(1,2).contiguous().reshape(b,t,d)
        output=self.output(joined)
        return (output,weights) if return_weights else output


class PositionwiseFeedForward(nn.Module):
    def __init__(self,dimensions,multiplier=4):
        super().__init__()
        if type(multiplier) is not int or multiplier<1:
            raise ValueError('Positive integer expansion multiplier required.')
        self.expand=nn.Linear(dimensions,multiplier*dimensions)
        self.activation=nn.GELU()
        self.contract=nn.Linear(multiplier*dimensions,dimensions)

    def forward(self,states):
        return self.contract(self.activation(self.expand(states)))


class DecoderBlock(nn.Module):
    """Pre-normalized attention and feature processing with residual paths."""
    def __init__(self,dimensions,heads,multiplier=4,dropout=0.0):
        super().__init__()
        self.attention_norm=nn.LayerNorm(dimensions,eps=1e-5)
        self.attention=MultiHeadCausalAttention(dimensions,heads,dropout)
        self.feedforward_norm=nn.LayerNorm(dimensions,eps=1e-5)
        self.feedforward=PositionwiseFeedForward(dimensions,multiplier)
        self.attention_dropout=nn.Dropout(dropout)
        self.feedforward_dropout=nn.Dropout(dropout)

    def forward(self,states,valid):
        states=states+self.attention_dropout(self.attention(self.attention_norm(states),valid))
        states=states+self.feedforward_dropout(self.feedforward(self.feedforward_norm(states)))
        return states*valid.unsqueeze(-1)
