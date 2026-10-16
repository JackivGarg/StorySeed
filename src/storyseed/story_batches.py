"""Compact story storage and nonoverlapping, padded, boundary-aware windows."""
import json
import numpy as np
import torch
from torch.nn import functional as F

from storyseed.corpus import PROCESSED, MANIFESTS, write_json, sha256, record
from storyseed.story_splits import load_split
from storyseed.story_tokenizer import StoryTokenizer, TOKENIZER_PATH

CONTEXT = 64


class DocumentBatches:
    def __init__(self, split: str, context: int = CONTEXT):
        if split not in {'train', 'development'} or context < 1:
            raise ValueError('Phase 2 batches support train/development and positive context only.')
        metadata = json.loads((MANIFESTS / 'encoded.json').read_text(encoding='utf-8'))
        if sha256(TOKENIZER_PATH.read_bytes()) != metadata['tokenizer_sha256']:
            raise ValueError('Encoded corpus and tokenizer disagree.')
        for suffix in ('tokens', 'offsets'):
            path = PROCESSED / f'{split}_{suffix}.npy'
            if sha256(path.read_bytes()) != metadata['files'][split][suffix + '_sha256']:
                raise ValueError('Encoded arrays differ from the saved manifest.')
        self.ids = np.load(PROCESSED / f'{split}_tokens.npy', mmap_mode='r')
        self.offsets = np.load(PROCESSED / f'{split}_offsets.npy', mmap_mode='r')
        self.context = context
        self.locations = [(document, start) for document in range(len(self.offsets)-1)
            for start in range(0, int(self.offsets[document+1]-self.offsets[document])-1, context)]

    def __len__(self):
        return len(self.locations)

    def batch(self, rows):
        rows = list(rows)
        if not rows or any(type(i) is not int or not 0 <= i < len(self) for i in rows):
            raise ValueError('Batch indices must be nonempty valid integer rows.')
        x = torch.zeros((len(rows), self.context), dtype=torch.long)
        y = torch.zeros_like(x)
        mask = torch.zeros_like(x, dtype=torch.bool)
        locations = []
        for b, row in enumerate(rows):
            document, start = self.locations[row]
            begin, end = int(self.offsets[document]), int(self.offsets[document+1])
            sequence = self.ids[begin:end]
            size = min(self.context, len(sequence)-1-start)
            x[b, :size] = torch.tensor(sequence[start:start+size].astype(np.int64))
            y[b, :size] = torch.tensor(sequence[start+1:start+size+1].astype(np.int64))
            mask[b, :size] = True
            locations.append((document, start))
        return {'inputs': x, 'targets': y, 'loss_mask': mask,
                'attention_mask': x != StoryTokenizer.pad_id, 'locations': locations}


def masked_next_token_loss(logits, targets, mask):
    if logits.ndim != 3 or logits.shape[:2] != targets.shape or targets.shape != mask.shape:
        raise ValueError('Expected scores [B,T,V] and labels/mask [B,T].')
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError('A boolean mask with at least one prediction is required.')
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='none')
    return losses.reshape_as(targets)[mask].mean()


def prepare_encoded():
    torch.set_num_threads(1)
    tokenizer = StoryTokenizer.load()
    metadata = {'tokenizer_sha256': sha256(TOKENIZER_PATH.read_bytes()),
        'context_demonstration': CONTEXT, 'files': {},
        'packing': 'one story per row chunk; nonoverlapping target positions; right-pad final chunk',
        'controls': 'one BOS before content and one EOS after; PAD only fills tensor shape',
        'final_holdout_access': 'not encoded; left reserved for later protocol'}
    for name in ('train', 'development'):
        rows = load_split(name)
        ids, offsets = [], [0]
        for row in rows:
            sequence = tokenizer.encode(row['text'], add_bos=True, add_eos=True)
            assert sequence[0] == tokenizer.bos_id and sequence[-1] == tokenizer.eos_id
            ids.extend(sequence)
            offsets.append(len(ids))
        dtype = np.uint16 if tokenizer.vocab_size <= 65536 else np.uint32
        token_path, offset_path = PROCESSED / f'{name}_tokens.npy', PROCESSED / f'{name}_offsets.npy'
        np.save(token_path, np.array(ids, dtype=dtype))
        np.save(offset_path, np.array(offsets, dtype=np.int64))
        metadata['files'][name] = {'stories': len(rows), 'stored_ids': len(ids),
            'prediction_positions': len(ids)-len(rows), 'dtype': str(np.dtype(dtype)),
            'tokens_sha256': sha256(token_path.read_bytes()), 'offsets_sha256': sha256(offset_path.read_bytes()),
            'source_split_sha256': sha256((PROCESSED / (name + '.jsonl')).read_bytes())}
    write_json(MANIFESTS / 'encoded.json', metadata)
    observations = {}
    for name in ('train', 'development'):
        dataset = DocumentBatches(name)
        valid = eos = 0
        for start in range(0, len(dataset), 128):
            batch = dataset.batch(list(range(start, min(len(dataset), start+128))))
            valid += int(batch['loss_mask'].sum())
            eos += int(((batch['targets'] == tokenizer.eos_id) & batch['loss_mask']).sum())
        assert valid == metadata['files'][name]['prediction_positions']
        assert eos == metadata['files'][name]['stories']
        observations[name] = {'chunks': len(dataset), 'scored_positions': valid, 'eos_targets': eos}
    source = DocumentBatches('train')
    short = next(i for i, (doc, start) in enumerate(source.locations)
                 if int(source.offsets[doc+1]-source.offsets[doc])-1-start < CONTEXT)
    demo = source.batch([0, short, len(source)-1])
    z = torch.zeros(3, CONTEXT, tokenizer.vocab_size)
    value = masked_next_token_loss(z, demo['targets'], demo['loss_mask'])
    altered = z.clone()
    altered[~demo['loss_mask']] = torch.arange(tokenizer.vocab_size).float() * 100
    assert torch.equal(value, masked_next_token_loss(altered, demo['targets'], demo['loss_mask']))
    record('05_batches', {'command': '.venv/Scripts/python scripts/phase02/05_prepare_batches.py',
        'context': CONTEXT, 'splits': observations, 'sample_shape': list(demo['inputs'].shape),
        'sample_valid_per_row': demo['loss_mask'].sum(1).tolist(),
        'padding_logits_do_not_change_loss': True, 'uniform_fixture_loss': value.item(),
        'prediction_weighting': 'all true target positions once; no mean-of-unequal-batch-means',
        'boundary_policy': metadata['packing'], 'final_holdout_encoded': False})
