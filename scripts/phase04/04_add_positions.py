"""Expose single-layer last-query order ambiguity and positional information."""
import torch
from storyseed.position_features import TokenPositionEmbedding
from storyseed.attention import CausalSelfAttention
from storyseed.attention_demo import projection_fixture
from storyseed.phase04_evidence import record


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    features = TokenPositionEmbedding(32,4,8)
    attention = CausalSelfAttention(4,2)
    _, fixture_projection = projection_fixture()
    attention.projection.load_state_dict(fixture_projection.state_dict())
    a = torch.tensor([[1,7,8,9]],dtype=torch.long)
    b = torch.tensor([[1,8,7,9]],dtype=torch.long)
    valid = a != 0
    no_position_a = attention(features.token(a),valid,last_only=True)
    no_position_b = attention(features.token(b),valid,last_only=True)
    with_position_a = attention(features(a),valid,last_only=True)
    with_position_b = attention(features(b),valid,last_only=True)
    assert torch.allclose(no_position_a,no_position_b,atol=1e-6,rtol=0)
    padded = torch.tensor([[0,0,1,7]],dtype=torch.long)
    short = torch.tensor([[1,7]],dtype=torch.long)
    padded_result = attention(features(padded),padded != 0,last_only=True)
    short_result = attention(features(short),short != 0,last_only=True)
    assert torch.allclose(padded_result,short_result,atol=1e-6,rtol=0)
    assert features(padded)[:,:2].count_nonzero() == 0
    record('04_positions', {
        'command': '.venv/Scripts/python scripts/phase04/04_add_positions.py',
        'fixture': 'synthetic IDs, seed-17 random token/position features and hand-set QKV projections; not trained language behavior',
        'original_ids': a.tolist(), 'reordered_earlier_ids': b.tolist(),
        'without_positions_max_last_output_difference': (no_position_a-no_position_b).abs().max().item(),
        'with_positions_max_last_output_difference': (with_position_a-with_position_b).abs().max().item(),
        'padded_ids': padded.tolist(), 'padded_position_ids': features.position_ids(padded).tolist(),
        'unpadded_position_ids': features.position_ids(short).tolist(),
        'padding_does_not_shift_real_position_features_max_output_difference': (padded_result-short_result).abs().max().item(),
        'padded_states_are_zero': True, 'position_parameter_shape': list(features.position.weight.shape),
        'position_policy': '0-based real-token offsets inside supplied window; recentered after history truncation',
        'claim_scope': 'single attention layer final query; not a claim that all position-free causal stacks are fully order invariant',
        'final_holdout_accessed': False})


if __name__ == '__main__':
    main()
