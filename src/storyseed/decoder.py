"""An explicitly composed pre-normalized GPT-style causal decoder."""
from dataclasses import dataclass,asdict
import torch
from torch import nn
from storyseed.position_features import TokenPositionEmbedding
from storyseed.decoder_components import DecoderBlock


@dataclass(frozen=True)
class DecoderConfig:
    vocabulary:int=512
    context:int=64
    dimensions:int=32
    heads:int=4
    layers:int=2
    multiplier:int=4
    dropout:float=0.1

    def __post_init__(self):
        for value in [self.vocabulary,self.context,self.dimensions,self.heads,self.layers,self.multiplier]:
            if type(value) is not int or value<1:raise ValueError('Positive integer decoder dimensions required.')
        if self.vocabulary<3 or self.dimensions%self.heads or not 0<=self.dropout<1:
            raise ValueError('Invalid vocabulary, head division or dropout probability.')

    def to_dict(self):return asdict(self)


class DecoderLanguageModel(nn.Module):
    def __init__(self,config:DecoderConfig):
        super().__init__();self.config=config
        self.features=TokenPositionEmbedding(config.vocabulary,config.dimensions,config.context)
        self.embedding_dropout=nn.Dropout(config.dropout)
        self.blocks=nn.ModuleList([DecoderBlock(config.dimensions,config.heads,config.multiplier,config.dropout) for _ in range(config.layers)])
        self.final_norm=nn.LayerNorm(config.dimensions,eps=1e-5)
        self.output=nn.Linear(config.dimensions,config.vocabulary,bias=False)
        self.apply(self._initialize)
        with torch.no_grad():self.features.token.lookup.weight[0].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if isinstance(module,nn.Linear) and module.bias is not None:nn.init.zeros_(module.bias)

    def hidden_states(self,ids):
        valid=ids!=0
        states=self.embedding_dropout(self.features(ids))
        for block in self.blocks:states=block(states,valid)
        return self.final_norm(states)*valid.unsqueeze(-1)

    def forward(self,ids):return self.output(self.hidden_states(ids))

    def forward_contexts(self,contexts):
        if contexts.ndim!=2 or contexts.shape[1]<1 or (contexts[:,-1]==0).any():
            raise ValueError('Contexts must end in a real current token.')
        return self.output(self.hidden_states(contexts)[:,-1])


def parameter_ledger(model):
    return {'token':model.features.token.lookup.weight.numel(),
            'position':model.features.position.weight.numel(),
            'blocks':[sum(p.numel() for p in block.parameters()) for block in model.blocks],
            'final_norm':sum(p.numel() for p in model.final_norm.parameters()),
            'vocabulary_head':sum(p.numel() for p in model.output.parameters()),
            'total':sum(p.numel() for p in model.parameters())}
