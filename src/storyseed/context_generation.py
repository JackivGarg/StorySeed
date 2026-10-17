"""A small common prediction interface for bigram and fixed-window models."""
import math
import torch
from storyseed.bigram import BigramLanguageModel


def predict_contexts(model, contexts):
    if isinstance(model, BigramLanguageModel):
        return model(contexts[:, -1:])[:, 0, :]
    return model.forward_contexts(contexts)


def context_width(model):
    return 1 if isinstance(model, BigramLanguageModel) else model.config.context


def history_context(model, history):
    width = context_width(model)
    selected = history[-width:]
    return torch.tensor([[0]*(width-len(selected)) + selected], dtype=torch.long)


@torch.no_grad()
def sample_context(model, tokenizer, prompt='Once upon a time', *, seed=17,
                   max_new_tokens=48, temperature=0.8, top_k=40):
    if not math.isfinite(temperature) or temperature <= 0 or max_new_tokens < 0:
        raise ValueError('Positive finite temperature and nonnegative token budget required.')
    if top_k is not None and not 1 <= top_k <= tokenizer.vocab_size-2:
        raise ValueError('top_k is outside ordinary/EOS choices.')
    model.eval()
    history = tokenizer.encode(prompt, add_bos=True)
    generator = torch.Generator().manual_seed(seed)
    generated, stop = [], 'length'
    for _ in range(max_new_tokens):
        scores = predict_contexts(model, history_context(model, history))[0].float().clone()/temperature
        scores[:2] = float('-inf')
        if top_k is not None:
            values, indices = scores.topk(top_k)
            scores = torch.full_like(scores, float('-inf')).scatter(0, indices, values)
        token = torch.multinomial(scores.softmax(-1), 1, generator=generator).item()
        generated.append(token)
        if token == tokenizer.eos_id:
            stop = 'eos'
            break
        history.append(token)
    return {'prompt': prompt, 'text': prompt+tokenizer.decode(generated),
            'generated_ids': generated, 'stop_reason': stop, 'seed': seed,
            'temperature': temperature, 'top_k': top_k,
            'context': context_width(model), 'max_new_tokens': max_new_tokens,
            'policy': 'forbid PAD/BOS while sampling; full vocabulary remains in evaluation loss'}
