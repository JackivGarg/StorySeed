"""A separately declared continuation after the first recovery still confused stories."""
from dataclasses import replace
import ctypes
import json
import time
import torch

from storyseed.corpus import ROOT, sha256
from storyseed.decoder import parameter_ledger
from storyseed.decoder_io import save_inference
from storyseed.experiment_data import write_json
from storyseed.experiment_training import generate
from storyseed.quality_data import RecoveryBatches, TOKENIZER, identity, plan
from storyseed.quality_training import configuration
from storyseed.story_tokenizer import StoryTokenizer
from storyseed.training_checkpoint import load_training, save_training
from storyseed.training_evaluation import evaluate_decoder

PROTOCOL = ROOT / 'planning/QUALITY_RECOVERY_CONTINUATION_PROTOCOL.json'
REPORTS = ROOT / 'reports/quality_recovery'
RUN = ROOT / 'runs/quality_recovery_continued'
CHECKPOINT = ROOT / 'checkpoints/quality_recovery_continued/latest.pt'


def train():
    if (ROOT / 'planning/QUALITY_RECOVERY_CANDIDATE.json').exists() or (REPORTS / '04_metrics.json').exists():
        raise ValueError('Final evaluation was already released. This pre-final continuation is no longer valid.')
    spec = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    parent = json.loads((REPORTS / '03_training.json').read_text(encoding='utf-8'))
    if parent['successful_updates'] != 16000 or parent['checkpoint']['sha256'] != spec['parent_checkpoint_sha256']:
        raise ValueError('Continuation parent differs from the declared completed run.')
    config = replace(configuration(), total_updates=spec['additional_updates'],
                     warmup_updates=spec['schedule']['warmup_updates'], learning_rate=spec['schedule']['learning_rate'],
                     minimum_learning_rate=spec['schedule']['minimum_learning_rate'])
    data, development = RecoveryBatches('train'), RecoveryBatches('development')
    base_identity = identity()
    data_id = dict(base_identity, continuation_protocol_sha256=sha256(PROTOCOL.read_bytes()),
                   parent_checkpoint_sha256=spec['parent_checkpoint_sha256'])
    tokenizer = StoryTokenizer.load(TOKENIZER, expected_sha256=base_identity['tokenizer_sha256'])
    RUN.mkdir(parents=True, exist_ok=True)
    history_path = RUN / 'history.json'
    outcome = REPORTS / '03b_training.json'
    if outcome.exists():
        raise ValueError('Completed continuation exists; do not rerun it.')
    if CHECKPOINT.exists():
        runner = load_training(CHECKPOINT, config, data=data, identity=data_id)
        history = json.loads(history_path.read_text(encoding='utf-8'))
        boundary = 16000 + runner.updates
        history['updates'] = [row for row in history['updates'] if row['update'] <= boundary]
        history['evaluations'] = [row for row in history['evaluations'] if row['update'] <= boundary]
        if history['best_update'] > boundary:
            raise RuntimeError('Best export is newer than the saved resume boundary; inspect.')
        prior_elapsed = history['stage_elapsed_seconds']
    else:
        parent_path = ROOT / 'checkpoints/quality_recovery/latest.pt'
        if sha256(parent_path.read_bytes()) != spec['parent_checkpoint_sha256']:
            raise ValueError('Parent checkpoint bytes changed.')
        runner = load_training(parent_path, configuration(), data=data, identity=base_identity)
        # Preserve parameters, AdamW moments/steps, scaler, RNG and row generator.
        # Only the explicitly declared stage counter/configuration/identity change.
        runner.config = config
        runner.data_identity = data_id
        runner.updates = 0
        history = {'configuration': config.to_dict(), 'identity': data_id, 'updates': [], 'evaluations': [],
                   'best_development_nll': parent['best_development_nll'], 'best_update': parent['best_update'],
                   'best_inference': parent['inference'], 'stage_elapsed_seconds': 0.0,
                   'initial_global_targets': runner.optimizer_target_tokens,
                   'parent_optimizer_attempts': runner.attempts,
                   'scope': 'declared new learning-rate stage; full-state continuity, not the unmodified initial schedule'}
        prior_elapsed = 0.0
    started = time.monotonic()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    def persist():
        history['stage_elapsed_seconds'] = prior_elapsed + time.monotonic() - started
        temp = history_path.with_suffix('.json.tmp')
        write_json(temp, history)
        temp.replace(history_path)
        artifact = save_training(runner, CHECKPOINT)
        write_json(REPORTS / 'LIVE_STATUS.json', {'state': 'continuation_training',
                   'updates': 16000 + runner.updates, 'total_updates': 16000 + config.total_updates,
                   'stage_updates': runner.updates, 'targets_presented': runner.optimizer_target_tokens,
                   'stage_elapsed_seconds': history['stage_elapsed_seconds'], 'best_update': history['best_update'],
                   'best_development_nll': history['best_development_nll'], 'checkpoint': artifact,
                   'fresh_final_accessed': False})
        return artifact

    def evaluate():
        metrics = evaluate_decoder(runner.model, development, config.evaluation_batch_rows)
        event = {'update': 16000 + runner.updates, 'stage_update': runner.updates, 'development': metrics}
        if metrics['nats_per_target_token'] < history['best_development_nll']:
            history['best_inference'] = save_inference(runner.model,
                ROOT / 'checkpoints/quality_recovery_continued/best-inference.pt',
                'quality recovery continuation; selected by fixed development only', tokenizer_path=TOKENIZER)
            history['best_development_nll'] = metrics['nats_per_target_token']
            history['best_update'] = event['update']
        if runner.updates % 4000 == 0:
            event['samples'] = [generate(runner.model, tokenizer, prompt, context=256, max_new_tokens=160)
                                for prompt in plan()['diagnostic_prompts'][:3]]
        history['evaluations'].append(event)
        print(json.dumps({'evaluation_update': event['update'], 'development_nll': metrics['nats_per_target_token'],
                          'best_update': history['best_update']}), flush=True)
        original_curve = json.loads((REPORTS / '03_curve.json').read_text(encoding='utf-8'))
        write_json(REPORTS / '03b_curve.json', {'initial_configuration': parent['configuration'],
                   'continuation_configuration': config.to_dict(), 'stage_boundary_update': 16000,
                   'updates': original_curve['updates'] + history['updates'],
                   'evaluations': parent['development_curve'] + history['evaluations']})

    try:
        persist()
        print(json.dumps({'continuation_started_at_global_update': 16000 + runner.updates,
                          'additional_budget': config.total_updates, 'final_population_reserved': True}), flush=True)
        while runner.updates < config.total_updates:
            if prior_elapsed + time.monotonic() - started > spec['maximum_stage_wall_seconds']:
                persist()
                raise RuntimeError('Continuation wall-time bound reached; full checkpoint saved.')
            torch.cuda.synchronize()
            tick = time.perf_counter()
            event = runner.step()
            torch.cuda.synchronize()
            event.pop('rows')
            event['step_seconds'] = time.perf_counter() - tick
            event['stage_update'] = event['update']
            event['update'] += 16000
            if event['optimizer_step_skipped']:
                raise RuntimeError('Skipped optimizer step; inspect before proceeding.')
            history['updates'].append(event)
            if runner.updates % 100 == 0:
                print(json.dumps({'update': event['update'], 'sampled_nll': event['nll'],
                                  'targets_presented': runner.optimizer_target_tokens,
                                  'stage_elapsed_seconds': prior_elapsed + time.monotonic() - started}), flush=True)
            if runner.updates % 1000 == 0:
                evaluate()
            if runner.updates % 250 == 0:
                persist()
        checkpoint = persist()
        result = {'status': 'complete', 'configuration': parent['configuration'],
                  'continuation_configuration': config.to_dict(), 'identity': data_id,
                  'successful_updates': 16000 + runner.updates, 'stage_successful_updates': runner.updates,
                  'parameter_ledger': parameter_ledger(runner.model),
                  'targets_presented': runner.optimizer_target_tokens,
                  'stage_targets_presented': runner.optimizer_target_tokens - history['initial_global_targets'],
                  'available_training_targets': sum(data.target_counts),
                  'target_presentations_per_pool': runner.optimizer_target_tokens / sum(data.target_counts),
                  'elapsed_seconds': parent['elapsed_seconds'] + history['stage_elapsed_seconds'],
                  'stage_elapsed_seconds': history['stage_elapsed_seconds'],
                  'best_update': history['best_update'], 'best_development_nll': history['best_development_nll'],
                  'inference': history['best_inference'], 'checkpoint': checkpoint,
                  'development_curve': parent['development_curve'] + history['evaluations'],
                  'fresh_final_accessed': False, 'continuation_protocol_sha256': sha256(PROTOCOL.read_bytes()),
                  'selection': 'minimum fixed-development NLL across initial and declared continuation stages',
                  'learner_understanding': 'pending'}
        write_json(outcome, result)
        write_json(REPORTS / 'LIVE_STATUS.json', {'state': 'training_complete_evaluation_pending',
                   'updates': result['successful_updates'], 'best_update': result['best_update'],
                   'best_development_nll': result['best_development_nll'], 'fresh_final_accessed': False})
        print(json.dumps({'continuation_complete': True, 'best_update': result['best_update'],
                          'best_development_nll': result['best_development_nll']}), flush=True)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
