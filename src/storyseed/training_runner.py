"""One correctly weighted effective-batch update at a time."""
import math
import random
import numpy as np
from contextlib import nullcontext
import torch
from storyseed.decoder import DecoderLanguageModel
from storyseed.training_config import data_identity
from storyseed.training_data import TrainingBatches,token_losses
from storyseed.training_optimizer import make_optimizer,set_update_learning_rate


class TrainingRunner:
    def __init__(self,config,*,data=None,identity=None):
        if config.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA was requested but is unavailable.')
        if config.precision=='bf16' and not torch.cuda.is_bf16_supported():raise ValueError('CUDA bf16 is unsupported on this machine.')
        self.config=config;self.device=torch.device(config.device)
        torch.set_num_threads(config.threads);torch.manual_seed(config.seed)
        random.seed(config.seed);np.random.seed(config.seed%(2**32))
        self.model=DecoderLanguageModel(config.model).to(self.device)
        self.optimizer=make_optimizer(self.model,config)
        self.scaler=torch.amp.GradScaler('cuda',enabled=config.precision=='fp16',init_scale=1024.0)
        self.data=data if data is not None else TrainingBatches('train',config.model.context,config.score_tokens)
        self.data_identity=identity if identity is not None else data_identity()
        self.data_generator=torch.Generator().manual_seed(config.data_seed)
        self.updates=0;self.attempts=0;self.tokens_consumed=0;self.optimizer_target_tokens=0
        self.at_boundary=True

    def autocast(self):
        if self.config.precision=='fp32':return nullcontext()
        return torch.autocast('cuda',dtype=torch.float16 if self.config.precision=='fp16' else torch.bfloat16)

    def step(self,rows=None):
        config=self.config
        if self.updates>=config.total_updates:raise ValueError('Declared update budget completed.')
        if not self.at_boundary:raise RuntimeError('Restore the last complete checkpoint after an interrupted attempt.')
        self.at_boundary=False
        if rows is None:
            rows=torch.randint(len(self.data),(config.micro_batch_rows*config.accumulation_steps,),generator=self.data_generator).tolist()
        else:rows=list(rows)
        if not rows:raise ValueError('An effective batch needs real rows.')
        batches=[self.data.batch(rows[start:start+config.micro_batch_rows]) for start in range(0,len(rows),config.micro_batch_rows)]
        counts=[int(batch['loss_mask'].sum()) for batch in batches]
        total=sum(counts);inputs=sum(int(batch['attention_mask'].sum()) for batch in batches)
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        lr=set_update_learning_rate(self.optimizer,config,self.updates+1)
        losses=[]
        for batch in batches:
            x,y,mask=[batch[key].to(self.device) for key in ['inputs','targets','loss_mask']]
            with self.autocast():
                values=token_losses(self.model(x),y,mask)
                objective=values.sum()/total
            if not torch.isfinite(objective):raise RuntimeError('Nonfinite microbatch loss.')
            self.scaler.scale(objective).backward();losses.append(values.detach().sum())
        self.scaler.unscale_(self.optimizer)
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),config.clip_norm,error_if_nonfinite=False).item()
        previous_scale=self.scaler.get_scale()
        if not math.isfinite(norm):
            if not self.scaler.is_enabled():raise RuntimeError('Nonfinite unscaled gradient norm.')
            self.scaler.update(new_scale=previous_scale/2)
            skipped=True
        else:
            self.scaler.step(self.optimizer);self.scaler.update()
            skipped=self.scaler.get_scale()<previous_scale
        self.attempts+=1;self.tokens_consumed+=total
        if not skipped:self.updates+=1;self.optimizer_target_tokens+=total
        self.at_boundary=True
        return {'update':self.updates,'attempt':self.attempts,'learning_rate':lr,'nll':torch.stack(losses).sum().item()/total,
                'true_targets':total,'input_tokens':inputs,'microbatch_target_counts':counts,'gradient_norm_before_clip':norm if math.isfinite(norm) else None,
                'optimizer_step_skipped':skipped,'loss_scale':self.scaler.get_scale(),'rows':rows}
