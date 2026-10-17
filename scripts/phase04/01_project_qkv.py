"""Trace the distinct roles of three projections of the same input states."""
import torch
from storyseed.attention_demo import projection_fixture
from storyseed.phase04_evidence import record


def main():
    torch.set_num_threads(1)
    features, projection = projection_fixture()
    q, k, v = projection(features)
    (q.sum()+k.sum()+v.sum()).backward()
    record('01_qkv', {
        'command': '.venv/Scripts/python scripts/phase04/01_project_qkv.py',
        'fixture': 'hand-set features and projection matrices; no training or linguistic semantics',
        'features': features.tolist(), 'query': q.detach().tolist(),
        'key': k.detach().tolist(), 'value': v.detach().tolist(),
        'input_shape': list(features.shape), 'projection_shapes': [list(x.shape) for x in [q,k,v]],
        'stored_parameters': sum(p.numel() for p in projection.parameters()),
        'gradient_probe': 'sum of Q, K and V outputs; not next-token cross-entropy',
        'projection_gradient_norms': {name: layer.weight.grad.norm().item()
              for name, layer in [('query',projection.query),('key',projection.key),('value',projection.value)]},
        'final_holdout_accessed': False})


if __name__ == '__main__':
    main()
