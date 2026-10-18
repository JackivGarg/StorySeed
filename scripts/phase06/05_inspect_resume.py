import copy,json,subprocess,sys
import torch
from storyseed.corpus import ROOT,write_json
from storyseed.training_config import TrainingConfig
from storyseed.training_runner import TrainingRunner
from storyseed.training_checkpoint import save_training,load_training,cpu_tree
from storyseed.build_evidence import record


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    config=TrainingConfig()
    full=TrainingRunner(config);history=[full.step() for _ in range(8)]
    expected_weights=cpu_tree(full.model.state_dict());expected_optimizer=cpu_tree(full.optimizer.state_dict())
    expected_data_state=full.data_generator.get_state();expected_rng=torch.get_rng_state().clone()
    interrupted=TrainingRunner(config)
    for _ in range(3):interrupted.step()
    checkpoint=ROOT/'checkpoints/phase06/resume-after-3.pt'
    save_training(interrupted,checkpoint)
    worker_checkpoint=ROOT/'checkpoints/phase06/resumed-after-8.pt'
    worker_report=ROOT/'runs/phase06/resume-worker.json'
    subprocess.run([sys.executable,str(ROOT/'scripts/train_decoder.py'),'--resume',str(checkpoint),
        '--updates','5','--checkpoint',str(worker_checkpoint),'--report',str(worker_report),'--max-seconds','25'],
        cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=45,check=True)
    result=json.loads(worker_report.read_text(encoding='utf-8'))
    resumed=load_training(worker_checkpoint,config)
    weights_equal=same(expected_weights,cpu_tree(resumed.model.state_dict()))
    optimizer_equal=same(expected_optimizer,cpu_tree(resumed.optimizer.state_dict()))
    rng_equal=torch.equal(expected_rng,torch.get_rng_state())
    sampler_equal=torch.equal(expected_data_state,resumed.data_generator.get_state())
    assert weights_equal and optimizer_equal and rng_equal and sampler_equal
    assert result['history']==history[3:]
    wrong=TrainingRunner(config);wrong.model.load_state_dict(interrupted.model.state_dict());wrong.updates=3
    for _ in range(5):wrong.step()
    weights_only_delta=max((expected_weights[name]-value.detach().cpu()).abs().max().item() for name,value in wrong.model.state_dict().items())
    assert weights_only_delta>1e-6
    artifact=result['checkpoint'];write_json(ROOT/'data/manifests/phase06/resume-checkpoint.json',artifact)
    record(6,'05_resume',{'command':'.venv/Scripts/python scripts/phase06/05_inspect_resume.py',
       'control':'CPU fp32, one thread, dropout 0.1, identical config and verified data',
       'comparison':'8 uninterrupted successful updates vs 3 + checkpoint + 5 updates in a fresh CLI process',
       'model_weights_bitwise_equal':weights_equal,'optimizer_state_bitwise_equal':optimizer_equal,
       'torch_rng_state_equal':rng_equal,'data_generator_state_equal':sampler_equal,
       'remaining_row_loss_lr_history_equal':True,'weights_only_continuation_max_weight_difference':weights_only_delta,
       'completed_updates':resumed.updates,'attempts':resumed.attempts,'consumed_target_tokens':resumed.tokens_consumed,
       'checkpoint':artifact,'scope':'same-environment CPU demonstration; cross-device/version bitwise identity is not claimed',
       'final_holdout_accessed':False})

if __name__=='__main__':main()
