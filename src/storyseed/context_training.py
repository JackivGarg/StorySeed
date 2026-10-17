"""A deliberately small fixed-budget neural-context training experiment."""
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from storyseed.corpus import ROOT, MANIFESTS, sha256, write_json
from storyseed.story_tokenizer import StoryTokenizer, TOKENIZER_PATH
from storyseed.fixed_context import ContextConfig, FixedWindowLanguageModel, CausalWindows
from storyseed.phase03_evidence import record

CONFIG_PATH = ROOT / 'configs/phase03_pilot.json'


def check_deadline(deadline):
    if deadline is not None and time.monotonic() > deadline:
        raise RuntimeError('Pilot compute deadline reached; no completed-result claim is valid.')


@torch.no_grad()
def evaluate_scores(predict, dataset, *, batch_size=2048, deadline=None):
    """Aggregate total negative log probability, not unequal batch means."""
    total = 0.0
    for start in range(0, len(dataset), batch_size):
        check_deadline(deadline)
        x, y = dataset.batch(np.arange(start, min(start+batch_size, len(dataset))))
        scores = predict(x)
        value = F.cross_entropy(scores.double(), y, reduction='sum').item()
        if not math.isfinite(value):
            raise RuntimeError('Nonfinite evaluation loss.')
        total += value
    mean = total / len(dataset)
    return {'nats_per_target_token': mean, 'perplexity': math.exp(mean),
            'prediction_positions': len(dataset)}


def load_context_model(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != 'storyseed.phase03.context.v1':
        raise ValueError('Unsupported neural-context checkpoint.')
    tokenizer = StoryTokenizer.load(expected_sha256=payload['tokenizer_sha256'])
    config = ContextConfig(**payload['model_configuration'])
    if config.vocabulary != tokenizer.vocab_size:
        raise ValueError('Checkpoint vocabulary differs from tokenizer.')
    model = FixedWindowLanguageModel(config)
    model.load_state_dict(payload['state_dict'], strict=True)
    if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Expected finite float32 model parameters.')
    if torch.count_nonzero(model.embedding.lookup.weight[0]):
        raise ValueError('Checkpoint padding row must remain zero.')
    return model.eval(), tokenizer


def train_condition(context):
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if context not in config['contexts']:
        raise ValueError('Context must be one of the declared pilot conditions.')
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['initialization_seed'])
    tokenizer = StoryTokenizer.load()
    train = CausalWindows('train', context)
    dev = CausalWindows('development', context)
    model = FixedWindowLanguageModel(ContextConfig(tokenizer.vocab_size, context,
                                                  config['dimensions'], config['hidden']))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    rng = np.random.default_rng(config['sampling_seed'])
    start = time.monotonic()
    deadline = start + config['compute_deadline_seconds_per_condition']
    initial = {name: evaluate_scores(model.forward_contexts, dataset,
               batch_size=config['evaluation_batch_targets'], deadline=deadline)
               for name, dataset in [('train', train), ('development', dev)]}
    history = []
    for step in range(1, config['steps']+1):
        check_deadline(deadline)
        rows = rng.integers(0, len(train), size=config['batch_targets'])
        x, y = train.batch(rows)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.forward_contexts(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss.')
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            history.append({'step': step, 'pre_update_minibatch_loss': loss.item()})
    model.eval()
    final = {name: evaluate_scores(model.forward_contexts, dataset,
             batch_size=config['evaluation_batch_targets'], deadline=deadline)
             for name, dataset in [('train', train), ('development', dev)]}
    assert final['train']['nats_per_target_token'] < initial['train']['nats_per_target_token']
    assert torch.count_nonzero(model.embedding.lookup.weight[0]) == 0
    path = ROOT / f'checkpoints/phase03/context_k{context}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format': 'storyseed.phase03.context.v1',
               'model_configuration': model.config.to_dict(),
               'tokenizer_sha256': sha256(TOKENIZER_PATH.read_bytes()),
               'encoded_manifest_sha256': sha256((MANIFESTS/'encoded.json').read_bytes()),
               'training_configuration': config,
               'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    restored, _ = load_context_model(path)
    probe, _ = train.batch([0, 64, len(train)-1])
    with torch.no_grad():
        assert torch.equal(model.forward_contexts(probe), restored.forward_contexts(probe))
    artifact = {'checkpoint': str(path.relative_to(ROOT)).replace('\\','/'),
                'sha256': sha256(path.read_bytes()), 'format': payload['format'],
                'tokenizer_sha256': payload['tokenizer_sha256'],
                'encoded_manifest_sha256': payload['encoded_manifest_sha256'],
                'inference_only': True, 'optimizer_resume': False}
    write_json(ROOT/f'data/manifests/phase03/context_k{context}.json', artifact)
    record(f'03_training_k{context}', {
        'command': f'.venv/Scripts/python scripts/phase03/03_train_context.py --context {context}',
        'configuration': config, 'model_configuration': model.config.to_dict(),
        'stored_parameters_including_fixed_pad_row': sum(p.numel() for p in model.parameters()),
        'sampled_training_targets_with_replacement': config['steps']*config['batch_targets'],
        'sampling': 'uniform over training target events, with replacement; same index RNG for both conditions',
        'initial': initial, 'final': final, 'minibatch_history': history,
        'elapsed_compute_seconds': time.monotonic()-start,
        'restored_probe_logits_equal': True, 'checkpoint': artifact,
        'final_holdout_accessed': False,
        'limits': 'single-seed fixed-budget demonstration; not a matched-parameter or matched-compute architecture study'})
