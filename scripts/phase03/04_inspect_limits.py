"""Observe a fixed information horizon and compare unchanged target populations."""
import json
import numpy as np
import torch
from storyseed.corpus import ROOT
from storyseed.context_training import load_context_model, evaluate_scores
from storyseed.context_generation import predict_contexts, history_context, sample_context
from storyseed.fixed_context import CausalWindows
from storyseed.story_baseline import load_baseline
from storyseed.phase03_evidence import record


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    model, tokenizer = load_context_model(ROOT/'checkpoints/phase03/context_k8.pt')
    suffix = [350, 387, 377, 407, 14, 282, 314, 85]
    a, b = [1, 17]+suffix, [1, 19]+suffix
    za = predict_contexts(model, history_context(model, a))
    zb = predict_contexts(model, history_context(model, b))
    changed = suffix.copy()
    changed[0] = 19
    zc = predict_contexts(model, history_context(model, changed))
    assert torch.equal(za, zb)
    sequence = torch.tensor([[1]+suffix+[11, 12, 13]], dtype=torch.long)
    altered = sequence.clone()
    altered[:, 6:] = 23
    original_scores, altered_scores = model(sequence), model(altered)
    assert torch.equal(original_scores[:, :6], altered_scores[:, :6])
    direct = torch.stack([model.forward_contexts(history_context(model, sequence[0, :t+1].tolist()))[0]
                          for t in range(sequence.shape[1])])
    delta = (direct-original_scores[0]).abs().max().item()
    assert delta < 1e-5
    bigram, _ = load_baseline()
    k1, _ = load_context_model(ROOT/'checkpoints/phase03/context_k1.pt')
    models = {'phase02_bigram': bigram, 'context_k1': k1, 'context_k8': model}
    metrics, samples = {}, {}
    for name, predictor in models.items():
        data = CausalWindows('development', 1 if name == 'phase02_bigram' else predictor.config.context)
        metrics[name] = evaluate_scores(lambda x: predict_contexts(predictor, x), data)
        samples[name] = sample_context(predictor, tokenizer)
    old = json.loads((ROOT/'reports/phase02/06_baseline.json').read_text(encoding='utf-8'))
    expected = old['history'][-1]['development']['nats_per_target_token']
    assert abs(metrics['phase02_bigram']['nats_per_target_token']-expected) < 1e-6
    record('04_limits', {
        'command': '.venv/Scripts/python scripts/phase03/04_inspect_limits.py',
        'shared_suffix_ids': suffix, 'outside_window_prefixes': [[1,17], [1,19]],
        'outside_window_change_produces_identical_logits': True,
        'inside_window_change_max_logit_difference': (za-zc).abs().max().item(),
        'changing_future_inputs_leaves_earlier_logits_unchanged': True,
        'sequence_forward_matches_prefix_last_position_max_difference': delta,
        'development_metrics': metrics, 'same_target_count': 39711,
        'phase02_report_reproduced_in_new_evaluator': True,
        'samples_same_prompt_seed_temperature_top_k_budget': samples,
        'sample_scope': 'illustrations, not blinded quality evaluation or final held-out cases',
        'final_holdout_accessed': False,
        'next_need': 'content-dependent retrieval over a wider causal prefix; attention is introduced in Phase 4'})


if __name__ == '__main__':
    main()
