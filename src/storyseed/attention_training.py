"""A bounded learning pilot for the explicit single-head attention predictor."""
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from storyseed.attention_model import AttentionConfig, AttentionLanguageModel
from storyseed.context_training import evaluate_scores, check_deadline
from storyseed.corpus import ROOT, MANIFESTS, sha256, write_json
from storyseed.fixed_context import CausalWindows
from storyseed.story_tokenizer import StoryTokenizer, TOKENIZER_PATH
from storyseed.phase04_evidence import record

CHECKPOINT = ROOT/'checkpoints/phase04/attention.pt'


def load_attention_model(path=CHECKPOINT):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != 'storyseed.phase04.attention.v1':
        raise ValueError('Unsupported attention checkpoint format.')
    config = AttentionConfig(**payload['model_configuration'])
    tokenizer = StoryTokenizer.load(expected_sha256=payload['tokenizer_sha256'])
    if config.vocabulary != tokenizer.vocab_size:
        raise ValueError('Checkpoint and tokenizer vocabularies differ.')
    for value in payload['state_dict'].values():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Expected finite float32 attention parameters.')
    model = AttentionLanguageModel(config)
    model.load_state_dict(payload['state_dict'], strict=True)
    if model.features.token.lookup.weight[0].count_nonzero():
        raise ValueError('Checkpoint padding token row must remain zero.')
    return model.eval(), tokenizer


def train_attention():
    protocol = json.loads((ROOT/'configs/phase04_pilot.json').read_text(encoding='utf-8'))
    torch.set_num_threads(protocol['threads'])
    torch.manual_seed(protocol['initialization_seed'])
    tokenizer = StoryTokenizer.load()
    config = AttentionConfig(tokenizer.vocab_size, protocol['context'], protocol['dimensions'],
                             protocol['head_dimensions'])
    train, dev = CausalWindows('train',config.context), CausalWindows('development',config.context)
    model = AttentionLanguageModel(config)
    optimizer = torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'])
    rng = np.random.default_rng(protocol['sampling_seed'])
    started = time.monotonic()
    deadline = started + protocol['compute_deadline_seconds']
    initial = {name: evaluate_scores(model.forward_contexts,data,
               batch_size=protocol['evaluation_batch_targets'],deadline=deadline)
               for name,data in [('train',train),('development',dev)]}
    history = []
    for step in range(1,protocol['steps']+1):
        check_deadline(deadline)
        rows = rng.integers(0,len(train),size=protocol['batch_targets'])
        x,y = train.batch(rows)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.forward_contexts(x),y)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite attention training loss.')
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            history.append({'step':step,'pre_update_minibatch_loss':loss.item()})
    model.eval()
    final = {name: evaluate_scores(model.forward_contexts,data,
             batch_size=protocol['evaluation_batch_targets'],deadline=deadline)
             for name,data in [('train',train),('development',dev)]}
    assert final['train']['nats_per_target_token'] < initial['train']['nats_per_target_token']
    CHECKPOINT.parent.mkdir(parents=True,exist_ok=True)
    payload = {'format':'storyseed.phase04.attention.v1','model_configuration':config.to_dict(),
               'training_configuration':protocol,'tokenizer_sha256':sha256(TOKENIZER_PATH.read_bytes()),
               'encoded_manifest_sha256':sha256((MANIFESTS/'encoded.json').read_bytes()),
               'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
    temporary = CHECKPOINT.with_suffix('.pt.tmp')
    torch.save(payload,temporary)
    temporary.replace(CHECKPOINT)
    restored,_ = load_attention_model()
    probe,_ = train.batch([0,64,len(train)-1])
    with torch.no_grad():
        assert torch.equal(model.forward_contexts(probe),restored.forward_contexts(probe))
    artifact = {'checkpoint':'checkpoints/phase04/attention.pt','format':payload['format'],
                'sha256':sha256(CHECKPOINT.read_bytes()),'tokenizer_sha256':payload['tokenizer_sha256'],
                'encoded_manifest_sha256':payload['encoded_manifest_sha256'],
                'inference_only':True,'optimizer_resume':False}
    write_json(ROOT/'data/manifests/phase04/attention.json',artifact)
    record('05_training',{
        'command':'.venv/Scripts/python scripts/phase04/05_train_attention.py',
        'protocol':protocol,'model_configuration':config.to_dict(),
        'stored_parameters_including_fixed_pad_row':sum(p.numel() for p in model.parameters()),
        'sampled_training_targets_with_replacement':protocol['steps']*protocol['batch_targets'],
        'initial':initial,'final':final,'minibatch_history':history,
        'elapsed_compute_seconds':time.monotonic()-started,'restored_probe_logits_equal':True,
        'checkpoint':artifact,'final_holdout_accessed':False,
        'architecture_scope':'one attention head plus token/position lookup and vocabulary head; no residual, LayerNorm, FFN, dropout or decoder stack',
        'comparison_limits':'same tokenizer/splits/target population; different architecture/capacity/compute from prior baselines'})
