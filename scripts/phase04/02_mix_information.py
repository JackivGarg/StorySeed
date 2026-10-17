"""Trace scaled pair matches into position weights and returned content."""
import torch
from storyseed.attention import attention_scores, scaled_attention
from storyseed.attention_demo import projection_fixture
from storyseed.phase04_evidence import record


def main():
    torch.set_num_threads(1)
    features, projection = projection_fixture()
    q, k, v = projection(features)
    output, weights = scaled_attention(q,k,v)
    changed_values = v.detach().clone()
    changed_values[:,2] += 10
    altered, same_weights = scaled_attention(q,k,changed_values)
    assert torch.equal(weights, same_weights)
    assert torch.allclose(weights.sum(-1), torch.ones(1,3))
    assert torch.allclose(output[0,2], torch.tensor([4/3,1.]))
    output.square().mean().backward()
    record('02_retrieval', {
        'command': '.venv/Scripts/python scripts/phase04/02_mix_information.py',
        'fixture': 'same hand-set Q/K/V illustration as 4.1; deliberately unmasked',
        'raw_pair_scores': (q.detach() @ k.detach().transpose(-2,-1)).tolist(),
        'scaled_scores': attention_scores(q.detach(),k.detach()).tolist(),
        'weights': weights.detach().tolist(), 'row_weight_sums': weights.detach().sum(-1).tolist(),
        'returned_values': output.detach().tolist(), 'output_shape': list(output.shape),
        'changing_only_values_preserves_weights': True,
        'changing_future_value_changes_first_output_by': (altered-output).abs()[0,0].max().item(),
        'projection_gradient_norms': {name: layer.weight.grad.norm().item()
             for name, layer in [('query',projection.query),('key',projection.key),('value',projection.value)]},
        'limitation': 'future positions are accessible; this illustrative function is not a causal predictor',
        'final_holdout_accessed': False})


if __name__ == '__main__':
    main()
