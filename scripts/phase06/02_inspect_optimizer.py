import torch
from storyseed.corpus import ROOT
from storyseed.training_config import TrainingConfig
from storyseed.training_optimizer import make_optimizer,set_update_learning_rate,learning_rate_for_update
from storyseed.training_data import TrainingBatches,token_losses
from storyseed.decoder import DecoderLanguageModel
from storyseed.build_evidence import record


def main():
    config=TrainingConfig.load(ROOT/'configs/phase06_teaching.json')
    torch.set_num_threads(1);torch.manual_seed(config.seed)
    model=DecoderLanguageModel(config.model);optimizer=make_optimizer(model,config)
    data=TrainingBatches('train',config.model.context,config.score_tokens)
    batch=data.batch([0,len(data)-1])
    values=token_losses(model(batch['inputs']),batch['targets'],batch['loss_mask'])
    before=model.output.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True);values.mean().backward()
    pre=torch.nn.utils.clip_grad_norm_(model.parameters(),config.clip_norm,error_if_nonfinite=True).item()
    post=sum(p.grad.double().square().sum() for p in model.parameters() if p.grad is not None).sqrt().item()
    lr=set_update_learning_rate(optimizer,config,1);optimizer.step()
    assert post<=config.clip_norm+1e-6
    record(6,'02_optimizer',{'command':'.venv/Scripts/python scripts/phase06/02_inspect_optimizer.py',
       'optimizer':'AdamW foreach=False; matrices including token/position tables decay, biases/LayerNorm vectors do not',
       'decay_parameter_count':sum(p.numel() for p in optimizer.param_groups[0]['params']),
       'no_decay_parameter_count':sum(p.numel() for p in optimizer.param_groups[1]['params']),
       'schedule_lr_at_updates':{str(i):learning_rate_for_update(config,i) for i in [1,4,5,20]},
       'first_update_learning_rate':lr,'gradient_norm_before_clip':pre,'gradient_norm_after_clip':post,
       'output_weight_max_change':(before-model.output.weight).abs().max().item(),
       'step_index_semantics':'one-based successful optimizer updates; warmup/cosine is a pure function of config and count',
       'final_holdout_accessed':False})

if __name__=='__main__':main()
