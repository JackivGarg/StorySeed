"""One correctly weighted effective-batch update at a time."""
import math
import torch
from storyseed.decoder import DecoderLanguageModel
from storyseed.training_data import TrainingBatches,token_losses
from storyseed.training_optimizer import make_optimizer,set_update_learning_rate


class TrainingRunner:
    def __init__(self,config):
        if config.device!='cpu' or config.precision!='fp32':raise ValueError('This increment supports CPU fp32 updates.')
        self.config=config;self.device=torch.device(config.device)
        torch.set_num_threads(config.threads);torch.manual_seed(config.seed)
        self.model=DecoderLanguageModel(config.model).to(self.device)
        self.optimizer=make_optimizer(self.model,config)
        self.data=TrainingBatches('train',config.model.context,config.score_tokens)
        self.data_generator=torch.Generator().manual_seed(config.data_seed)
        self.updates=0;self.attempts=0;self.tokens_consumed=0;self.optimizer_target_tokens=0

    def step(self,rows=None):
        config=self.config
        if self.updates>=config.total_updates:raise ValueError('Declared update budget completed.')
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
            values=token_losses(self.model(x),y,mask)
            objective=values.sum()/total
            if not torch.isfinite(objective):raise RuntimeError('Nonfinite microbatch loss.')
            objective.backward();losses.append(values.detach().sum())
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),config.clip_norm,error_if_nonfinite=True).item()
        self.optimizer.step()
        self.attempts+=1;self.updates+=1;self.tokens_consumed+=total;self.optimizer_target_tokens+=total
        return {'update':self.updates,'attempt':self.attempts,'learning_rate':lr,'nll':torch.stack(losses).sum().item()/total,
                'true_targets':total,'input_tokens':inputs,'microbatch_target_counts':counts,'gradient_norm_before_clip':norm,
                'optimizer_step_skipped':False,'rows':rows}
