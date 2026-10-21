"""A recoverable, development-selected scratch run for the quality recovery."""
import json
import math
import statistics
import time
from pathlib import Path
import torch

from storyseed.corpus import ROOT, sha256
from storyseed.decoder import DecoderConfig, parameter_ledger
from storyseed.decoder_io import save_inference
from storyseed.experiment_data import write_json
from storyseed.experiment_training import generate
from storyseed.quality_data import plan, prepare, RecoveryBatches, MANIFEST, TOKENIZER, identity
from storyseed.story_tokenizer import StoryTokenizer
from storyseed.training_config import TrainingConfig
from storyseed.training_runner import TrainingRunner
from storyseed.training_checkpoint import save_training, load_training
from storyseed.training_evaluation import evaluate_decoder

REPORTS = ROOT / 'reports/quality_recovery'
RUN = ROOT / 'runs/quality_recovery'
CHECKPOINT = ROOT / 'checkpoints/quality_recovery/latest.pt'


def configuration():
    spec = plan()
    values = dict(spec['training'])
    values['total_updates'] = values.pop('initial_total_updates')
    return TrainingConfig(model=DecoderConfig(**spec['model']), **values)


def pilot():
    config = configuration()
    data = RecoveryBatches('train')
    runner = TrainingRunner(config, data=data, identity=identity())
    durations = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(16):
        torch.cuda.synchronize()
        started = time.perf_counter()
        event = runner.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    measured = durations[4:]
    report = {'purpose': 'discarded scratch resource pilot; not the main learning run',
              'configuration': config.to_dict(), 'parameters': parameter_ledger(runner.model),
              'warmup_steps': 4, 'timed_steps': len(measured), 'all_step_seconds': durations,
              'median_seconds_per_step': statistics.median(measured),
              'rough_16000_update_hours_excluding_evaluation': statistics.median(measured) * config.total_updates / 3600,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'device': torch.cuda.get_device_name(), 'torch': str(torch.__version__), 'main_training_started': False}
    write_json(REPORTS / '02_resource_pilot.json', report)
    print(json.dumps(report, indent=2), flush=True)


def train():
    prepare()
    spec, config = plan(), configuration()
    data, development = RecoveryBatches('train'), RecoveryBatches('development')
    data_id = identity()
    tokenizer = StoryTokenizer.load(TOKENIZER, expected_sha256=data_id['tokenizer_sha256'])
    RUN.mkdir(parents=True, exist_ok=True)
    history_path = RUN / 'history.json'
    outcome_path = REPORTS / '03_training.json'
    if outcome_path.exists():
        raise ValueError('Completed recovery result exists; do not silently rerun or overwrite it.')
    if CHECKPOINT.exists():
        runner = load_training(CHECKPOINT, config, data=data, identity=data_id)
        history = json.loads(history_path.read_text(encoding='utf-8'))
        history['updates'] = [event for event in history['updates'] if event['update'] <= runner.updates]
        history['evaluations'] = [event for event in history['evaluations'] if event['update'] <= runner.updates]
        if history['best_update'] > runner.updates:
            raise RuntimeError('Selected inference is newer than resume boundary; inspect before proceeding.')
        prior_elapsed = history['elapsed_seconds']
    else:
        runner = TrainingRunner(config, data=data, identity=data_id)
        history = {'configuration': config.to_dict(), 'identity': data_id, 'updates': [], 'evaluations': [],
                   'best_development_nll': None, 'best_update': 0, 'best_inference': None, 'elapsed_seconds': 0.0}
        prior_elapsed = 0.0
    started = time.monotonic()
    # Keep the machine awake for this finite run; do not change persistent power settings.
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    def persist():
        history['elapsed_seconds'] = prior_elapsed + time.monotonic() - started
        temporary = history_path.with_suffix('.json.tmp')
        write_json(temporary, history)
        temporary.replace(history_path)
        artifact = save_training(runner, CHECKPOINT)
        write_json(REPORTS / 'LIVE_STATUS.json', {'state': 'training', 'updates': runner.updates,
                   'total_updates': config.total_updates, 'targets_presented': runner.optimizer_target_tokens,
                   'elapsed_seconds': history['elapsed_seconds'], 'best_update': history['best_update'],
                   'best_development_nll': history['best_development_nll'], 'checkpoint': artifact})
        return artifact

    def evaluate():
        metrics = evaluate_decoder(runner.model, development, config.evaluation_batch_rows)
        event = {'update': runner.updates, 'development': metrics}
        if runner.updates > 0 and (history['best_development_nll'] is None or metrics['nats_per_target_token'] < history['best_development_nll']):
            history['best_inference'] = save_inference(runner.model, ROOT / 'checkpoints/quality_recovery/best-inference.pt',
                                'quality recovery; selected only by fixed new development NLL', tokenizer_path=TOKENIZER)
            history['best_development_nll'] = metrics['nats_per_target_token']
            history['best_update'] = runner.updates
        if runner.updates % 2000 == 0:
            event['samples'] = [generate(runner.model, tokenizer, prompt, context=256, max_new_tokens=160)
                                for prompt in spec['diagnostic_prompts'][:3]]
        history['evaluations'].append(event)
        print(json.dumps({'evaluation_update': runner.updates, 'development_nll': metrics['nats_per_target_token'],
                          'best_update': history['best_update']}, ensure_ascii=True), flush=True)
        write_json(REPORTS / '03_curve.json', {'configuration': config.to_dict(),
                   'updates': history['updates'], 'evaluations': history['evaluations']})

    try:
        if not history['evaluations']:
            evaluate()
        persist()
        while runner.updates < config.total_updates:
            if prior_elapsed + time.monotonic() - started > spec['maximum_training_wall_seconds']:
                persist()
                raise RuntimeError('Recovery reached its declared wall-time allowance; checkpoint saved, completion not claimed.')
            torch.cuda.synchronize()
            tick = time.perf_counter()
            event = runner.step()
            torch.cuda.synchronize()
            event.pop('rows')
            event['step_seconds'] = time.perf_counter() - tick
            if event['optimizer_step_skipped']:
                raise RuntimeError('Recovery optimizer step skipped; inspect before continuation.')
            history['updates'].append(event)
            if runner.updates % 100 == 0:
                print(json.dumps({'update': runner.updates, 'sampled_nll': event['nll'],
                                  'targets_presented': runner.optimizer_target_tokens,
                                  'elapsed_seconds': prior_elapsed + time.monotonic() - started}), flush=True)
            if runner.updates % spec['evaluation_every'] == 0:
                evaluate()
            if runner.updates % spec['checkpoint_every'] == 0:
                persist()
        checkpoint = persist()
        result = {'status': 'complete', 'configuration': config.to_dict(), 'identity': data_id,
                  'successful_updates': runner.updates, 'parameter_ledger': parameter_ledger(runner.model),
                  'targets_presented': runner.optimizer_target_tokens, 'available_training_targets': sum(data.target_counts),
                  'target_presentations_per_pool': runner.optimizer_target_tokens / sum(data.target_counts),
                  'elapsed_seconds': history['elapsed_seconds'], 'best_update': history['best_update'],
                  'best_development_nll': history['best_development_nll'], 'inference': history['best_inference'],
                  'checkpoint': checkpoint, 'development_curve': history['evaluations'], 'fresh_final_accessed': False,
                  'selection': 'minimum fixed-development NLL; no final score selection', 'learner_understanding': 'pending'}
        write_json(outcome_path, result)
        write_json(REPORTS / 'LIVE_STATUS.json', {'state': 'training_complete_evaluation_pending', 'updates': runner.updates,
                                                'best_update': history['best_update'], 'best_development_nll': history['best_development_nll']})
        print(json.dumps({'training_complete': True, 'best_update': history['best_update'],
                          'best_development_nll': history['best_development_nll']}, ensure_ascii=True), flush=True)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
