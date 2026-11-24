"""A bounded post-release continuation; preserve the published model and evidence."""
from dataclasses import replace
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from storyseed.corpus import ROOT
from storyseed.decoder_io import load_inference, save_inference
from storyseed.experiment_training import generate
from storyseed.quality_data import RecoveryBatches, TOKENIZER
from storyseed.quality_evaluation import TextBatches
from storyseed.research_metrics import evaluate_by_story
from storyseed.story_tokenizer import StoryTokenizer
from storyseed.training_checkpoint import load_training, save_training
from storyseed.training_config import TrainingConfig
from storyseed.training_evaluation import evaluate_decoder

REPORT = ROOT / 'reports/followup_20261125'
RUN = ROOT / 'runs/followup_20261125'
WEIGHTS = ROOT / 'checkpoints/followup_20261125'
PROTOCOL = ROOT / 'planning/FOLLOWUP_20261125.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    if (REPORT / 'TRAINING.json').exists():
        raise ValueError('Completed follow-up exists; do not rerun.')
    lock = RUN / 'active.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    runner = None
    started = time.monotonic()
    try:
        spec = read(PROTOCOL)
        parent = read(ROOT / 'reports/quality_recovery/03b_training.json')
        parent_path = Path(parent['checkpoint']['path'])
        if digest(parent_path) != spec['parent_checkpoint_sha256']:
            raise ValueError('Parent checkpoint changed.')
        if digest(TOKENIZER) != parent['identity']['tokenizer_sha256']:
            raise ValueError('Tokenizer changed.')
        data, development = RecoveryBatches('train'), RecoveryBatches('development')
        config = replace(TrainingConfig.from_dict(parent['continuation_configuration']),
                         total_updates=spec['additional_updates'], warmup_updates=0,
                         learning_rate=0.00003, minimum_learning_rate=0.00001)
        data_id = dict(parent['identity'], followup_protocol_sha256=digest(PROTOCOL))
        history_path = RUN / 'history.json'
        checkpoint = WEIGHTS / 'latest.pt'
        tokenizer = StoryTokenizer.load(TOKENIZER, expected_sha256=parent['identity']['tokenizer_sha256'])
        if checkpoint.exists():
            runner = load_training(checkpoint, config, data=data, identity=data_id)
            history = read(history_path)
            history['updates'] = [x for x in history['updates'] if x['stage_update'] <= runner.updates]
            history['evaluations'] = [x for x in history['evaluations'] if x['stage_update'] <= runner.updates]
            if history['best_stage_update'] > runner.updates:
                raise RuntimeError('Best artifact is newer than resume boundary; inspect before resume.')
            elapsed_before = history['elapsed_seconds']
        else:
            runner = load_training(parent_path, TrainingConfig.from_dict(parent['continuation_configuration']),
                                   data=data, identity=parent['identity'])
            runner.config = config
            runner.data_identity = data_id
            runner.updates = 0
            history = {'updates': [], 'evaluations': [], 'best_stage_update': 0,
                       'best_nll': parent['best_development_nll'], 'best_inference': parent['inference'],
                       'initial_targets': runner.optimizer_target_tokens, 'elapsed_seconds': 0.0}
            elapsed_before = 0.0
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

        def persist(state='training'):
            history['elapsed_seconds'] = elapsed_before + time.monotonic() - started
            write(history_path, history)
            artifact = save_training(runner, checkpoint)
            write(REPORT / 'STATUS.json', {'state': state, 'pid': os.getpid(),
                  'stage_updates': runner.updates, 'global_updates': 48000 + runner.updates,
                  'stage_budget': 4000, 'best_stage_update': history['best_stage_update'],
                  'best_development_nll': history['best_nll'], 'checkpoint': artifact,
                  'elapsed_seconds': history['elapsed_seconds'], 'app_changed': False})
            return artifact

        if not history['evaluations']:
            baseline = evaluate_decoder(runner.model, development, 8)
            history['evaluations'].append({'stage_update': 0, 'global_update': 48000, 'development': baseline})
            history['best_nll'] = baseline['nats_per_target_token']
        persist()
        print(json.dumps({'started': True, 'pid': os.getpid(), 'additional_updates': 4000,
                          'baseline_nll': history['best_nll']}), flush=True)
        while runner.updates < config.total_updates:
            if elapsed_before + time.monotonic() - started > spec['maximum_training_seconds']:
                persist('stopped_wall_bound')
                raise RuntimeError('Declared wall bound reached; checkpoint saved.')
            torch.cuda.synchronize()
            tick = time.perf_counter()
            event = runner.step()
            torch.cuda.synchronize()
            event.pop('rows')
            event['stage_update'] = event.pop('update')
            event['global_update'] = 48000 + event['stage_update']
            event['step_seconds'] = time.perf_counter() - tick
            if event['optimizer_step_skipped']:
                raise RuntimeError('Skipped/nonfinite update; inspect.')
            history['updates'].append(event)
            if runner.updates % 500 == 0:
                result = evaluate_decoder(runner.model, development, 8)
                history['evaluations'].append({'stage_update': runner.updates,
                    'global_update': 48000 + runner.updates, 'development': result})
                if result['nats_per_target_token'] < history['best_nll']:
                    history['best_inference'] = save_inference(runner.model, WEIGHTS / 'best-inference.pt',
                        'post-release follow-up; development-selected only', tokenizer_path=TOKENIZER)
                    history['best_nll'] = result['nats_per_target_token']
                    history['best_stage_update'] = runner.updates
                print(json.dumps({'stage_update': runner.updates, 'development_nll': result['nats_per_target_token'],
                                  'best_nll': history['best_nll']}), flush=True)
            if runner.updates % 250 == 0:
                persist()
        full = persist('training_complete')
        outcome = {'status': 'complete', 'stage_updates': runner.updates, 'global_updates': 48000 + runner.updates,
                   'configuration': config.to_dict(), 'protocol_sha256': digest(PROTOCOL),
                   'best_stage_update': history['best_stage_update'], 'best_development_nll': history['best_nll'],
                   'inference': history['best_inference'], 'checkpoint': full,
                   'additional_target_presentations': runner.optimizer_target_tokens - history['initial_targets'],
                   'elapsed_seconds': history['elapsed_seconds'], 'evaluations': history['evaluations'],
                   'app_changed': False, 'human_ratings': 'pending', 'learner_understanding': 'pending'}
        write(REPORT / 'TRAINING.json', outcome)
        # Fix the candidate before opening/encoding new final story text.
        write(REPORT / 'CANDIDATE.json', {'inference': outcome['inference'], 'protocol_sha256': digest(PROTOCOL),
                                        'selected_stage_update': history['best_stage_update']})
        del runner
        runner = None
        torch.cuda.empty_cache()
        final_path = ROOT / spec['fresh_final_path']
        if digest(final_path) != spec['fresh_final_sha256']:
            raise ValueError('New final population changed.')
        rows = [json.loads(line) for line in final_path.read_text(encoding='utf-8').splitlines()]
        byte_count = sum(len(row['text'].encode('utf-8')) for row in rows)
        outcomes, samples = {}, []
        for name, path in [('current48000', Path(parent['inference']['path'])),
                           ('followup_selected', Path(outcome['inference']['path']))]:
            model, tok = load_inference(path, tokenizer_path=TOKENIZER)
            model.to('cuda')
            scores = evaluate_by_story(model, TextBatches(rows, tok, 256, 128), batch_rows=8)
            scores['population'] = 'new follow-up final 200 stories; same tokenizer/context for both fixed models'
            scores['bits_per_utf8_content_byte'] = sum(scores['story_loss_sums']) / (byte_count * np.log(2))
            outcomes[name] = scores
            for prompt in spec['prompts']:
                samples.append({'model': name, **generate(model, tok, prompt, context=256, seed=17,
                    max_new_tokens=192, temperature=0.8, top_k=40)})
            del model
            torch.cuda.empty_cache()
        write(REPORT / 'COMPARISON.json', {'models': outcomes, 'fresh_final_stories': len(rows),
              'fresh_final_now_used': True, 'scope': 'single-model training trajectory; selection uses development; one seed; qualitative review pending'})
        write(REPORT / 'SAMPLES.json', samples)
        write(REPORT / 'STATUS.json', {'state': 'complete_review_pending', 'stage_updates': 4000,
              'global_updates': 52000, 'best_stage_update': history['best_stage_update'],
              'best_development_nll': history['best_nll'], 'fresh_final_now_used': True,
              'app_changed': False, 'review': 'pending assistant/human review; no automatic promotion'})
        notes = Path(spec['notes_folder'])
        notes.mkdir(parents=True, exist_ok=True)
        table = '\n'.join(f"| {e['global_update']} | {e['development']['nats_per_target_token']:.6f} |" for e in history['evaluations'])
        (notes / '02-actual-learning.md').write_text('# What the extra training actually did\n\n'
            f"The additional 4,000 updates completed, presenting {outcome['additional_target_presentations']:,} scored targets. "
            'The fixed development objective uses the same 46,082 events as before. Changing sampled training losses use different rows and training dropout.\n\n'
            '| Global updates | Fixed-development NLL |\n|---:|---:|\n' + table + '\n\n'
            'The small cosine learning-rate schedule changed along with exposure. We did not isolate their individual effects. '
            'The selected artifact was fixed before scoring the new final population; that population is now used evidence. '
            'Read all raw continuations in reports/followup_20261125/SAMPLES.json before deciding whether story quality improved. '
            'A lower loss alone does not certify stable characters or objects. The app is unchanged pending review.\n', encoding='utf-8')
        print(json.dumps({'complete': True, 'best_nll': history['best_nll'], 'app_changed': False}), flush=True)
    except Exception as error:
        write(REPORT / 'ERROR.json', {'error': str(error), 'traceback': traceback.format_exc(), 'pid': os.getpid()})
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
