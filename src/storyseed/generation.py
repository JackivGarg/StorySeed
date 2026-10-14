"""Bounded, seeded autoregressive sampling from the Phase 1 bigram."""

import math

import torch

from storyseed.tokenizer import CharacterTokenizer


@torch.no_grad()
def generate(model, tokenizer: CharacterTokenizer, prompt: str, *, max_new_tokens: int = 64,
             seed: int = 17, temperature: float = 1.0, top_k: int | None = None) -> dict:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    if top_k is not None and not 1 <= top_k <= tokenizer.vocab_size:
        raise ValueError("top_k must be between 1 and the vocabulary size.")
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model and tokenizer vocabulary sizes differ.")
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("Provide at least one supported prompt character.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    generated = []
    reason = "length"
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            current = torch.tensor([[ids[-1]]], dtype=torch.long, device=model.scores.device)
            # Sampling is done on CPU so a saved CPU model can reproduce the
            # same sample from identical weights and a seeded CPU generator.
            logits = model(current)[0, 0].float().cpu() / temperature
            if top_k is not None:
                values, indices = logits.topk(top_k)
                restricted = torch.full_like(logits, float("-inf"))
                restricted[indices] = values
                logits = restricted
            probabilities = logits.softmax(dim=-1)
            next_id = torch.multinomial(probabilities, 1, generator=generator).item()
            generated.append(next_id)
            if next_id == tokenizer.eos_id:
                reason = "eos"
                break
            ids.append(next_id)
    finally:
        model.train(was_training)
    return {"prompt": prompt, "text": prompt + tokenizer.decode(generated),
            "generated_token_ids": generated, "sampled_tokens": len(generated),
            "stop_reason": reason, "seed": seed, "temperature": temperature, "top_k": top_k}
