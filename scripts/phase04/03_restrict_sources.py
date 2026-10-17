"""Trace causal/padding restrictions and an empty padded query row."""
import torch
from storyseed.attention import causal_attention
from storyseed.attention_demo import projection_fixture
from storyseed.phase04_evidence import record


def main():
    torch.set_num_threads(1)
    features, projection = projection_fixture()
    q, k, v = projection(features)
    valid = torch.ones(1,3,dtype=torch.bool)
    output, weights = causal_attention(q,k,v,valid)
    changed = features.clone()
    changed[:,1:] += 40
    other = causal_attention(*projection(changed),valid)[0]
    assert torch.equal(output[:,0],other[:,0])
    assert weights.triu(diagonal=1).count_nonzero() == 0
    probes = [x.detach().clone().requires_grad_() for x in (q,k,v)]
    causal_attention(*probes,valid)[0][:,0].square().sum().backward()
    future_gradients = {name: probe.grad[:,1:].abs().max().item()
                        for name,probe in zip(['query','key','value'],probes)}
    assert all(value == 0 for value in future_gradients.values())
    padded = torch.tensor([[False,False,True]])
    pad_output, pad_weights = causal_attention(q,k,v,padded)
    assert torch.isfinite(pad_output).all()
    assert torch.count_nonzero(pad_weights[:,:2]) == 0
    final_output, final_weights = causal_attention(q[:,-1:],k,v,valid)
    assert torch.equal(final_output,output[:,-1:])
    record('03_causality', {
        'command': '.venv/Scripts/python scripts/phase04/03_restrict_sources.py',
        'causal_weights': weights.detach().tolist(), 'returned_values': output.detach().tolist(),
        'future_weight_mass': weights.triu(diagonal=1).sum().item(),
        'future_input_change_max_first_output_difference': (output[:,0]-other[:,0]).abs().max().item(),
        'first_output_objective_future_gradients': future_gradients,
        'padding_valid_mask': padded.tolist(), 'padding_weights': pad_weights.detach().tolist(),
        'padding_row_sums': pad_weights.detach().sum(-1).tolist(),
        'padded_query_outputs_are_zero_and_finite': bool(torch.isfinite(pad_output).all() and not pad_output[:,:2].count_nonzero()),
        'last_query_only_equals_full_final_row': True,
        'mask_semantics': 'True means allowed real token; candidates must be at/before query and nonpadding',
        'final_holdout_accessed': False})


if __name__ == '__main__':
    main()
