from dataclasses import replace
import copy
import torch
from storyseed.training_config import TrainingConfig
from storyseed.training_runner import TrainingRunner
from storyseed.training_data import token_losses
from storyseed.build_evidence import record


def main():
    base=TrainingConfig(model=replace(TrainingConfig().model,dropout=0.0),clip_norm=1e6)
    whole=TrainingRunner(replace(base,micro_batch_rows=3,accumulation_steps=1))
    initial=copy.deepcopy(whole.model.state_dict())
    split=TrainingRunner(replace(base,micro_batch_rows=1,accumulation_steps=3))
    short=min(range(len(whole.data)),key=lambda row:whole.data.target_counts[row])
    rows=[0,short,1]
    whole_result=whole.step(rows);split_result=split.step(rows)
    gradient_delta=max((a.grad-b.grad).abs().max().item() for a,b in zip(whole.model.parameters(),split.model.parameters()))
    weight_delta=max((a-b).abs().max().item() for a,b in zip(whole.model.parameters(),split.model.parameters()))
    assert gradient_delta<2e-6 and weight_delta<2e-5
    wrong=TrainingRunner(replace(base,micro_batch_rows=1,accumulation_steps=3))
    wrong.model.load_state_dict(initial)
    wrong.optimizer.zero_grad(set_to_none=True)
    for row in rows:
        batch=wrong.data.batch([row])
        values=token_losses(wrong.model(batch['inputs']),batch['targets'],batch['loss_mask'])
        (values.mean()/len(rows)).backward()
    wrong_delta=max((a.grad-b.grad).abs().max().item() for a,b in zip(whole.model.parameters(),wrong.model.parameters()))
    assert wrong_delta>1e-5
    record(6,'03_accumulation',{'command':'.venv/Scripts/python scripts/phase06/03_inspect_accumulation.py',
       'selected_rows':rows,'microbatch_target_counts':split_result['microbatch_target_counts'],
       'global_target_denominator':split_result['true_targets'],'whole_vs_accumulated_max_gradient_difference':gradient_delta,
       'whole_vs_accumulated_max_updated_weight_difference':weight_delta,
       'wrong_mean_of_microbatch_means_gradient_difference':wrong_delta,
       'control':'CPU fp32, one thread, dropout disabled, same initial weights/rows, clipping inactive at 1e6',
       'normalization':'sum each microbatch token loss / total true targets across the effective batch',
       'final_holdout_accessed':False})

if __name__=='__main__':main()
