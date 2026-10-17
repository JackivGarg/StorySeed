"""Inspect lookup, equivalent one-hot multiplication and accumulated gradients."""
import torch
from storyseed.token_features import TokenEmbedding
from storyseed.story_tokenizer import StoryTokenizer
from storyseed.phase03_evidence import record


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    tokenizer = StoryTokenizer.load()
    embedding = TokenEmbedding(tokenizer.vocab_size, 32)
    ids = torch.tensor([[0, 1, 7, 7, 9]], dtype=torch.long)
    features = embedding(ids)
    one_hot = torch.nn.functional.one_hot(ids, tokenizer.vocab_size).float()
    dense = one_hot @ embedding.lookup.weight
    features.sum().backward()
    gradient = embedding.lookup.weight.grad
    assert torch.equal(features, dense)
    assert torch.equal(gradient[7], torch.full((32,), 2.0))
    assert torch.count_nonzero(gradient[0]) == 0
    assert torch.count_nonzero(gradient[8]) == 0
    record('01_embeddings', {
        'command': '.venv/Scripts/python scripts/phase03/01_inspect_embeddings.py',
        'seed': 17, 'vocabulary': tokenizer.vocab_size, 'embedding_dimensions': 32,
        'ids': ids.tolist(), 'input_shape': list(ids.shape), 'feature_shape': list(features.shape),
        'parameter_shape': list(embedding.lookup.weight.shape),
        'parameter_count': embedding.lookup.weight.numel(),
        'one_hot_multiplication_matches_lookup': True,
        'first_four_features_of_id_7': features[0, 2, :4].detach().tolist(),
        'loss_for_gradient_probe': 'sum of all output features; not the language-model objective',
        'gradient_first_coordinate': {str(i): gradient[i, 0].item() for i in [0, 1, 7, 8, 9]},
        'meaning': 'Repeated occurrences share one parameter row; their gradient contributions add.',
        'training_status': 'random features; no semantic or language-model learning claim',
        'final_holdout_accessed': False})


if __name__ == '__main__':
    main()
