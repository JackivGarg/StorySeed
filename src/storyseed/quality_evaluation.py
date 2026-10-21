"""Freeze one development-selected recovery artifact, then compare on new text."""
import json
import math
from pathlib import Path
import re
import numpy as np
import torch

from storyseed.corpus import ROOT, sha256, read_jsonl
from storyseed.decoder_io import load_inference
from storyseed.experiment_data import write_json
from storyseed.experiment_training import generate
from storyseed.quality_data import MANIFEST, FOLDER, TOKENIZER, PROTOCOL, plan
from storyseed.research_metrics import evaluate_by_story, continuation_metrics
from storyseed.training_data import TrainingBatches

REPORTS = ROOT / 'reports/quality_recovery'


class TextBatches(TrainingBatches):
    def __init__(self, rows, tokenizer, context, score_tokens):
        values, offsets = [], [0]
        for row in rows:
            values.extend(tokenizer.encode(row['text'], add_bos=True, add_eos=True))
            offsets.append(len(values))
        self.ids, self.offsets = np.asarray(values, dtype=np.uint16), np.asarray(offsets, dtype=np.int64)
        self.context, self.score_tokens = context, score_tokens
        self.locations = [(doc, start) for doc in range(len(rows)) for start in range(0, offsets[doc + 1] - offsets[doc] - 1, score_tokens)]
        self.target_counts = [min(score_tokens, offsets[doc + 1] - offsets[doc] - 1 - start) for doc, start in self.locations]


def evaluate():
    torch.set_num_threads(1)
    spec = plan()
    continuation_path = ROOT / 'planning/QUALITY_RECOVERY_CONTINUATION_PROTOCOL.json'
    expected_updates = spec['training']['initial_total_updates']
    training_path = REPORTS / '03_training.json'
    if continuation_path.exists():
        amendment = json.loads(continuation_path.read_text(encoding='utf-8'))
        expected_updates += amendment['additional_updates']
        training_path = REPORTS / '03b_training.json'
    training = json.loads(training_path.read_text(encoding='utf-8'))
    if training['successful_updates'] != expected_updates:
        raise ValueError('Full declared recovery budget is not complete.')
    candidate = Path(training['inference']['path'])
    if sha256(candidate.read_bytes()) != training['inference']['sha256']:
        raise ValueError('Selected inference changed after training.')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if sha256(TOKENIZER.read_bytes()) != manifest['tokenizer_sha256']:
        raise ValueError('Recovery tokenizer changed.')
    freeze_path = ROOT / 'planning/QUALITY_RECOVERY_CANDIDATE.json'
    frozen = {'format': 'storyseed.quality-candidate.v1', 'protocol_sha256': sha256(PROTOCOL.read_bytes()),
              'model': 'StorySeed-recovery4096', 'model_sha256': training['inference']['sha256'],
              'tokenizer_sha256': manifest['tokenizer_sha256'], 'selected_update': training['best_update'],
              'selection': 'minimum recorded fixed-development NLL, before fresh-final access', 'context': 256,
              'checkpoint': candidate.relative_to(ROOT).as_posix(), 'tokenizer': TOKENIZER.relative_to(ROOT).as_posix()}
    if continuation_path.exists():
        frozen['continuation_protocol_sha256'] = sha256(continuation_path.read_bytes())
        if training['continuation_protocol_sha256'] != frozen['continuation_protocol_sha256']:
            raise ValueError('Continuation result/protocol identity differs.')
    if freeze_path.exists():
        if json.loads(freeze_path.read_text(encoding='utf-8')) != frozen:
            raise ValueError('Another recovery candidate was already frozen.')
    else:
        write_json(freeze_path, frozen)
    result_path = REPORTS / '04_metrics.json'
    if result_path.exists():
        raise ValueError('Fresh-final evaluation already recorded; do not silently repeat selection/evaluation.')
    # No final text or encoded target is loaded before the fixed candidate above exists.
    final_path = FOLDER / 'fresh_final.jsonl'
    if sha256(final_path.read_bytes()) != manifest['fresh_final']['jsonl_sha256']:
        raise ValueError('New reserved stories changed.')
    rows = read_jsonl(final_path)
    content_bytes = sum(len(row['text'].encode('utf-8')) for row in rows)
    byte_counts = [len(row['text'].encode('utf-8')) for row in rows]
    model_specs = [('original512', ROOT / 'checkpoints/phase07/main128-inference.pt', ROOT / 'artifacts/phase02/tokenizer.json', 128, 64),
                   ('recovery4096', candidate, TOKENIZER, 256, 128)]
    outcomes, samples = {}, []
    for name, path, tokenizer_path, context, score in model_specs:
        model, tokenizer = load_inference(path, tokenizer_path=tokenizer_path)
        model.to('cuda')
        data = TextBatches(rows, tokenizer, context, score)
        metrics = evaluate_by_story(model, data, batch_rows=8)
        metrics['population'] = 'new recovery reserved stories, same text across tokenizers; original final set not rescored'
        metrics['bits_per_utf8_content_byte'] = sum(metrics['story_loss_sums']) / (content_bytes * math.log(2))
        metrics['normalization'] = 'all next-ID losses including EOS / UTF-8 content bytes; tokenizer/context differ'
        outcomes[name] = metrics
        for prompt_index, prompt in enumerate(spec['diagnostic_prompts']):
            for seed in spec['sampling']['seeds']:
                output = generate(model, tokenizer, prompt, context=context, seed=seed,
                                  temperature=spec['sampling']['temperature'], top_k=spec['sampling']['top_k'],
                                  max_new_tokens=spec['sampling']['max_new_tokens'])
                output.update({'model': name, 'prompt_index': prompt_index,
                               'surface_metrics': continuation_metrics(output['continuation'])})
                samples.append(output)
        del model
        torch.cuda.empty_cache()
    old, new = outcomes['original512'], outcomes['recovery4096']
    generator = np.random.default_rng(20261021)
    differences = []
    old_sums, new_sums, lengths = np.asarray(old['story_loss_sums']), np.asarray(new['story_loss_sums']), np.asarray(byte_counts)
    for _ in range(2000):
        picked = generator.integers(len(rows), size=len(rows))
        differences.append(float((new_sums[picked].sum() - old_sums[picked].sum()) / (lengths[picked].sum() * math.log(2))))
    report = {'frozen_candidate': frozen, 'final_stories': len(rows), 'utf8_content_bytes': content_bytes,
              'models': outcomes, 'new_minus_old_bits_per_byte': new['bits_per_utf8_content_byte'] - old['bits_per_utf8_content_byte'],
              'paired_story_bootstrap_95_percentile': np.quantile(differences, [0.025, 0.975]).tolist(),
              'scope': 'conditional on fixed models; combined data/tokenizer/context/training changes; not training-seed uncertainty or an isolated ablation',
              'fresh_final_now_used': True, 'human_ratings': 'pending'}
    write_json(REPORTS / '04_samples.json', {'sampling': spec['sampling'], 'samples': samples})
    write_json(result_path, report)
    lines = ['# Recovery continuation packet', '',
             'Identified assistant review is separate from pending human review. Full raw outputs/settings are in 04_samples.json.',
             'Each excerpt below shows at most 80 whitespace-separated continuation words. The 192-new-ID generation budget means different text lengths across tokenizers; excerpts are display-only, not rewritten model output.', '']
    for prompt_index, prompt in enumerate(spec['diagnostic_prompts']):
        for seed in spec['sampling']['seeds']:
            lines.extend([f'## Prompt {prompt_index + 1}, seed {seed}', '', prompt, ''])
            for name in ['original512', 'recovery4096']:
                sample = next(item for item in samples if item['prompt_index'] == prompt_index and item['seed'] == seed and item['model'] == name)
                excerpt = ' '.join(sample['continuation'].split()[:80])
                lines.extend([f'### {name}', '', excerpt, '', f"Stop: {sample['stop_reason']}; generated IDs: {len(sample['generated_ids'])}.", ''])
    (REPORTS / '04_SAMPLE_PACKET.md').write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'new_minus_old_bits_per_byte': report['new_minus_old_bits_per_byte'],
                      'old_bits_per_byte': old['bits_per_utf8_content_byte'], 'new_bits_per_byte': new['bits_per_utf8_content_byte'],
                      'retained_samples': len(samples), 'fresh_final_now_used': True}), flush=True)
    return report
