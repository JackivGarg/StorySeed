"""Trace an ordered context into features, hidden activations and scores."""
import torch
from storyseed.fixed_context import ContextConfig, FixedWindowLanguageModel, CausalWindows
from storyseed.phase03_evidence import record


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = FixedWindowLanguageModel(ContextConfig())
    data = CausalWindows('train', model.config.context)
    rows = [0, 64, len(data)-1]
    x, y = data.batch(rows)
    embedded = model.embedding(x)
    flat = embedded.flatten(1)
    hidden = torch.tanh(model.hidden(flat))
    logits = model.forward_contexts(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert x[0].tolist() == [0]*7 + [1]
    assert y[0].item() == int(data.ids[1])
    assert x[1].tolist() == data.ids[57:65].tolist()
    assert len(data) == 359933
    record('02_context', {
        'command': '.venv/Scripts/python scripts/phase03/02_inspect_context.py',
        'configuration': model.config.to_dict(), 'rows': rows,
        'contexts': x.tolist(), 'targets': y.tolist(),
        'shapes': {'contexts': list(x.shape), 'embedding': list(embedded.shape),
                   'flattened': list(flat.shape), 'hidden': list(hidden.shape), 'logits': list(logits.shape)},
        'stored_parameters_including_fixed_pad_row': sum(p.numel() for p in model.parameters()),
        'random_model_loss': loss.item(),
        'embedding_gradient_norm': model.embedding.lookup.weight.grad.norm().item(),
        'hidden_gradient_norm': model.hidden.weight.grad.norm().item(),
        'target_count': len(data), 'target_index_64_retains_prior_chunk_history': True,
        'context_order': 'oldest to current; PAD on the left only before the story begins',
        'objective': 'one real target per context; no padded targets in this adapter',
        'final_holdout_accessed': False})


if __name__ == '__main__':
    main()
