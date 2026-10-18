"""Full boundary checkpoints, including the state that selects the next update."""
import random
import numpy as np
import torch
from storyseed.corpus import sha256
from storyseed.training_config import TrainingConfig,data_identity
from storyseed.training_optimizer import learning_rate_for_update
from storyseed.training_runner import TrainingRunner


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def runtime_identity(runner):
    return {'torch':str(torch.__version__),'device':runner.config.device,'threads':runner.config.threads,
       'device_name':torch.cuda.get_device_name() if runner.config.device=='cuda' else 'cpu',
       'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
       'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
       'cuda_matmul_tf32':torch.backends.cuda.matmul.allow_tf32}


def random_state(runner):
    state=np.random.get_state()
    return {'python':random.getstate(),'numpy':{'algorithm':state[0],'keys':state[1].tolist(),
        'position':int(state[2]),'has_gaussian':int(state[3]),'cached_gaussian':float(state[4])},
        'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all() if runner.config.device=='cuda' else []}


def save_training(runner,path):
    if not runner.at_boundary:raise RuntimeError('Checkpoint only after a complete effective-batch attempt.')
    next_lr=learning_rate_for_update(runner.config,runner.updates+1) if runner.updates<runner.config.total_updates else None
    payload={'format':'storyseed.decoder.training.v1','configuration':runner.config.to_dict(),
       'data_identity':runner.data_identity,'runtime':runtime_identity(runner),'model':cpu_tree(runner.model.state_dict()),
       'optimizer':cpu_tree(runner.optimizer.state_dict()),'scaler':runner.scaler.state_dict(),
       'counters':{name:getattr(runner,name) for name in ['updates','attempts','tokens_consumed','optimizer_target_tokens']},
       'next_learning_rate':next_lr,'data_generator':runner.data_generator.get_state(),
       'random':random_state(runner),'model_training_mode':runner.model.training,
       'boundary':'after a complete effective-batch attempt; next step begins by clearing gradients'}
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pt.tmp');torch.save(payload,temporary);temporary.replace(path)
    return {'path':str(path),'sha256':sha256(path.read_bytes()),'format':payload['format'],'completed_updates':runner.updates}


def load_training(path,expected_config=None,*,data=None,identity=None):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format')!='storyseed.decoder.training.v1':raise ValueError('Expected full training checkpoint, not inference-only weights.')
    config=TrainingConfig.from_dict(payload['configuration'])
    if expected_config is not None and config.to_dict()!=expected_config.to_dict():raise ValueError('Resume configuration differs from the saved experiment.')
    if payload['data_identity']!=(identity if identity is not None else data_identity()):raise ValueError('Tokenizer or encoded data identity changed.')
    runner=TrainingRunner(config,data=data,identity=identity)
    if payload['runtime']!=runtime_identity(runner):raise ValueError('Training runtime differs; exact continuation is not assumed.')
    if any(value.dtype!=torch.float32 or not torch.isfinite(value).all() for value in payload['model'].values()):
        raise ValueError('Invalid saved master parameters.')
    runner.model.load_state_dict(payload['model'],strict=True)
    runner.optimizer.load_state_dict(payload['optimizer']);runner.scaler.load_state_dict(payload['scaler'])
    for name,value in payload['counters'].items():setattr(runner,name,value)
    if not 0<=runner.updates<=config.total_updates or runner.attempts<runner.updates:raise ValueError('Invalid progress counters.')
    next_lr=learning_rate_for_update(config,runner.updates+1) if runner.updates<config.total_updates else None
    if next_lr!=payload['next_learning_rate']:raise ValueError('Schedule progress is inconsistent.')
    runner.data_generator.set_state(payload['data_generator'])
    runner.model.train(payload['model_training_mode'])
    # Restore randomness last: construction initialized parameters and consumed RNG.
    state=payload['random'];random.setstate(state['python'])
    ns=state['numpy'];np.random.set_state((ns['algorithm'],np.asarray(ns['keys'],dtype=np.uint32),ns['position'],ns['has_gaussian'],ns['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if config.device=='cuda':torch.cuda.set_rng_state_all(state['torch_cuda'])
    return runner
