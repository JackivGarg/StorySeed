"""Strict run settings and identities for reproducible training."""
from dataclasses import dataclass,field,asdict
import json,math
from storyseed.decoder import DecoderConfig
from storyseed.corpus import ROOT,MANIFESTS,sha256
from storyseed.story_tokenizer import TOKENIZER_PATH


@dataclass(frozen=True)
class TrainingConfig:
    model:DecoderConfig=field(default_factory=DecoderConfig)
    score_tokens:int=32
    micro_batch_rows:int=4
    accumulation_steps:int=2
    total_updates:int=20
    warmup_updates:int=4
    learning_rate:float=0.003
    minimum_learning_rate:float=0.0003
    weight_decay:float=0.1
    beta1:float=0.9
    beta2:float=0.95
    clip_norm:float=1.0
    seed:int=17
    data_seed:int=2026
    device:str='cpu'
    precision:str='fp32'
    threads:int=1
    evaluation_batch_rows:int=16

    def __post_init__(self):
        integers=[self.score_tokens,self.micro_batch_rows,self.accumulation_steps,self.total_updates,self.threads,self.evaluation_batch_rows]
        if any(type(x) is not int or x<1 for x in integers):raise ValueError('Positive integer run counts required.')
        if type(self.warmup_updates) is not int or not 0<=self.warmup_updates<=self.total_updates:raise ValueError('Invalid warmup budget.')
        if self.score_tokens>self.model.context:raise ValueError('Scored block exceeds context.')
        scalars=[self.learning_rate,self.minimum_learning_rate,self.weight_decay,self.beta1,self.beta2,self.clip_norm]
        if not all(math.isfinite(x) for x in scalars):raise ValueError('Finite optimizer settings required.')
        if not 0<=self.minimum_learning_rate<=self.learning_rate or self.learning_rate<=0 or self.weight_decay<0 or self.clip_norm<=0:
            raise ValueError('Invalid learning rate, decay or clipping threshold.')
        if not 0<=self.beta1<1 or not 0<=self.beta2<1:raise ValueError('Invalid Adam moments.')
        if self.device not in {'cpu','cuda'} or self.precision not in {'fp32','fp16','bf16'}:raise ValueError('Unsupported device/precision.')
        if self.device=='cpu' and self.precision!='fp32':raise ValueError('This runner supports CPU fp32 only.')
        if type(self.seed) is not int or type(self.data_seed) is not int:raise ValueError('Integer seeds required.')

    def to_dict(self):return asdict(self)

    @classmethod
    def from_dict(cls,data):
        values=dict(data);values['model']=DecoderConfig(**values['model'])
        return cls(**values)

    @classmethod
    def load(cls,path):return cls.from_dict(json.loads(path.read_text(encoding='utf-8')))


def data_identity():
    return {'tokenizer_sha256':sha256(TOKENIZER_PATH.read_bytes()),'encoded_manifest_sha256':sha256((MANIFESTS/'encoded.json').read_bytes())}
