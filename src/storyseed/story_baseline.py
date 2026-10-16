"""Fit the common-tokenizer bigram objective exactly using transition counts."""
import json
import math
from pathlib import Path
import numpy as np
import torch

from storyseed.bigram import BigramLanguageModel
from storyseed.corpus import ROOT, PROCESSED, MANIFESTS, write_json, sha256, record
from storyseed.story_batches import DocumentBatches
from storyseed.story_tokenizer import StoryTokenizer, TOKENIZER_PATH

BASELINE_PATH = ROOT / 'checkpoints/phase02/bigram.pt'


def transition_counts(dataset: DocumentBatches, vocabulary: int):
    counts = np.zeros(vocabulary*vocabulary, dtype=np.int64)
    for begin, end in zip(dataset.offsets[:-1], dataset.offsets[1:]):
        ids = np.asarray(dataset.ids[int(begin):int(end)], dtype=np.int64)
        counts += np.bincount(ids[:-1]*vocabulary + ids[1:], minlength=vocabulary*vocabulary)
    return torch.tensor(counts.reshape(vocabulary, vocabulary), dtype=torch.float32)


def weighted_loss(scores, counts):
    logp = scores.log_softmax(dim=-1)
    return -(counts*logp).sum()/counts.sum()


@torch.no_grad()
def metrics(scores, counts):
    loss = weighted_loss(scores.double(), counts.double()).item()
    return {'nats_per_target_token': loss, 'perplexity': math.exp(loss), 'prediction_positions': int(counts.sum())}


@torch.no_grad()
def sample(model, tokenizer, prompt='Once upon a time', *, seed=17, max_new_tokens=48, temperature=1.0, top_k=None):
    if not math.isfinite(temperature) or temperature <= 0 or max_new_tokens < 0:
        raise ValueError('Positive finite temperature and nonnegative generation budget required.')
    if top_k is not None and not 1 <= top_k <= tokenizer.vocab_size-2:
        raise ValueError('top_k is outside the ordinary/EOS choice count.')
    history = tokenizer.encode(prompt, add_bos=True)
    generator = torch.Generator().manual_seed(seed)
    generated=[]; stop='length'
    for _ in range(max_new_tokens):
        logits = model.scores[history[-1]].detach().float().cpu()/temperature
        logits[:2] = float('-inf')
        if top_k is not None:
            values, indices = logits.topk(top_k)
            filtered = torch.full_like(logits, float('-inf'))
            filtered[indices] = values
            logits = filtered
        next_id = torch.multinomial(logits.softmax(-1), 1, generator=generator).item()
        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            stop='eos'; break
        history.append(next_id)
    return {'prompt': prompt, 'text': prompt+tokenizer.decode(generated), 'generated_ids': generated,
            'stop_reason': stop, 'seed': seed, 'temperature': temperature, 'top_k': top_k,
            'policy': 'PAD/BOS forbidden during sampling; evaluation uses the full vocabulary distribution'}


def load_baseline(path=BASELINE_PATH):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != 'storyseed.phase02.bigram.v1':
        raise ValueError('Unsupported Phase 2 baseline format.')
    tokenizer = StoryTokenizer.load(expected_sha256=payload['tokenizer_sha256'])
    scores = payload['state_dict'].get('scores')
    if not isinstance(scores, torch.Tensor) or scores.shape != (tokenizer.vocab_size, tokenizer.vocab_size):
        raise ValueError('Incompatible baseline table.')
    if scores.dtype != torch.float32 or not torch.isfinite(scores).all():
        raise ValueError('Expected finite float32 baseline scores.')
    model = BigramLanguageModel(tokenizer.vocab_size)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.eval(), tokenizer


def fit_common_baseline():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    tokenizer = StoryTokenizer.load()
    train = DocumentBatches('train')
    development = DocumentBatches('development')
    counts = transition_counts(train, tokenizer.vocab_size)
    dev_counts = transition_counts(development, tokenizer.vocab_size)
    assert int(counts.sum()) == len(train.ids)-len(train.offsets)+1
    model = BigramLanguageModel(tokenizer.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    history=[]
    steps=200
    for step in range(steps+1):
        if step==0 or step%50==0:
            history.append({'step':step,'train':metrics(model.scores,counts),'development':metrics(model.scores,dev_counts)})
        if step==steps:break
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_loss(model.scores,counts)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite baseline training loss.')
        loss.backward()
        optimizer.step()
    assert history[-1]['train']['nats_per_target_token'] < history[0]['train']['nats_per_target_token']
    alpha=0.1
    count_scores=(counts+alpha).log()
    unigram=(counts.sum(0)+alpha).log().unsqueeze(0).expand_as(counts)
    observed=counts>0
    empirical=counts/counts.sum(1,keepdim=True).clamp_min(1)
    entropy=-(counts[observed]*empirical[observed].log()).sum().item()/counts.sum().item()
    BASELINE_PATH.parent.mkdir(parents=True,exist_ok=True)
    payload={'format':'storyseed.phase02.bigram.v1','tokenizer_sha256':sha256(TOKENIZER_PATH.read_bytes()),
             'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
             'configuration':{'steps':steps,'learning_rate':0.1,'seed':17,'device':'cpu'}}
    temporary=BASELINE_PATH.with_suffix('.pt.tmp')
    torch.save(payload,temporary); temporary.replace(BASELINE_PATH)
    restored, restored_tokenizer=load_baseline()
    assert torch.equal(model.scores,restored.scores)
    expected=sample(model,tokenizer)
    assert expected==sample(restored,restored_tokenizer)
    manifest={'format':payload['format'],'checkpoint':'checkpoints/phase02/bigram.pt',
              'sha256':sha256(BASELINE_PATH.read_bytes()),'tokenizer_sha256':payload['tokenizer_sha256'],
              'inference_only':True,'optimizer_resume':False}
    write_json(MANIFESTS/'baseline.json',manifest)
    record('06_baseline',{'command':'.venv/Scripts/python scripts/phase02/06_fit_baseline.py',
        'configuration':payload['configuration'],'vocabulary':tokenizer.vocab_size,
        'trainable_parameters':model.scores.numel(),'history':history,
        'objective':'exact full-batch next-token cross-entropy, grouped by identical current/target IDs',
        'train_infimum_nats':entropy,'smoothed_count_alpha':alpha,
        'smoothed_unigram':{'train':metrics(unigram,counts),'development':metrics(unigram,dev_counts)},
        'smoothed_bigram':{'train':metrics(count_scores,counts),'development':metrics(count_scores,dev_counts)},
        'same_weights_and_seeded_output_after_reload':True,'sample':expected,
        'final_holdout_scored':False,'checkpoint':manifest,
        'comparison_scope':'future models must share tokenizer, splits, BOS/EOS and target positions; Phase 1 losses use different units'})
