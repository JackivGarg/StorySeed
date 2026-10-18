"""AdamW groups and a schedule indexed by successful optimizer updates."""
import math
import torch


def learning_rate_for_update(config,update):
    if type(update) is not int or not 1<=update<=config.total_updates:raise ValueError('Update outside declared schedule.')
    if config.warmup_updates and update<=config.warmup_updates:
        return config.learning_rate*update/config.warmup_updates
    if config.warmup_updates:
        fraction=(update-config.warmup_updates)/max(1,config.total_updates-config.warmup_updates)
    else:
        fraction=(update-1)/max(1,config.total_updates-1)
    return config.minimum_learning_rate+0.5*(config.learning_rate-config.minimum_learning_rate)*(1+math.cos(math.pi*fraction))


def make_optimizer(model,config):
    decay=[];no_decay=[]
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:(decay if parameter.ndim>=2 else no_decay).append(parameter)
    ids=[id(p) for p in decay+no_decay]
    assert len(ids)==len(set(ids))==sum(p.requires_grad for p in model.parameters())
    groups=[{'params':decay,'weight_decay':config.weight_decay},{'params':no_decay,'weight_decay':0.0}]
    return torch.optim.AdamW(groups,lr=config.learning_rate,betas=(config.beta1,config.beta2),eps=1e-8,foreach=False)


def set_update_learning_rate(optimizer,config,update):
    value=learning_rate_for_update(config,update)
    for group in optimizer.param_groups:group['lr']=value
    return value
