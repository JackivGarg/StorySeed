"""A transparent full-batch training loop for the Phase 1 score table."""

from dataclasses import asdict, dataclass
import math
import time

import torch

from storyseed.bigram import BigramLanguageModel
from storyseed.data import transition_pairs
from storyseed.loss import next_token_loss
from storyseed.tokenizer import CharacterTokenizer
from storyseed.toy_data import DEV_STORIES, TRAIN_STORIES


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 400
    learning_rate: float = 0.1
    seed: int = 17
    device: str = "auto"


def choose_device(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu or cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)


@torch.no_grad()
def evaluate(model, inputs, targets) -> float:
    was_training = model.training
    model.eval()
    value = next_token_loss(model(inputs), targets).item()
    model.train(was_training)
    return value


def empirical_distribution(inputs, targets, vocabulary: int):
    counts = torch.zeros(vocabulary, vocabulary, dtype=torch.float64)
    for current, following in zip(inputs.flatten().tolist(), targets.flatten().tolist()):
        counts[current, following] += 1
    probabilities = counts / counts.sum(dim=1, keepdim=True).clamp_min(1)
    seen = counts > 0
    entropy = -(counts[seen] * probabilities[seen].log()).sum().item() / counts.sum().item()
    return counts, probabilities, entropy


def train_bigram(config: TrainingConfig = TrainingConfig()):
    if config.steps < 1 or not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("Positive steps and finite positive learning rate are required.")
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    device = choose_device(config.device)
    tokenizer = CharacterTokenizer.fit(TRAIN_STORIES)
    train_x, train_y = transition_pairs(TRAIN_STORIES, tokenizer)
    dev_x, dev_y = transition_pairs(DEV_STORIES, tokenizer)
    counts, distribution, entropy = empirical_distribution(train_x, train_y, tokenizer.vocab_size)
    train_x, train_y = train_x.to(device), train_y.to(device)
    dev_x, dev_y = dev_x.to(device), dev_y.to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = BigramLanguageModel(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    initial_train = evaluate(model, train_x, train_y)
    initial_dev = evaluate(model, dev_x, dev_y)
    history = [{"step": 0, "train_loss": initial_train, "development_loss": initial_dev}]
    start = time.perf_counter()
    for step in range(1, config.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model(train_x), train_y)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite.")
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == config.steps:
            history.append({"step": step, "train_loss": evaluate(model, train_x, train_y), "development_loss": evaluate(model, dev_x, dev_y)})
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.eval()
    m_id = tokenizer.encode("m")[0]
    m_probabilities = model.scores[m_id].detach().softmax(dim=-1).cpu()
    result = {
        "configuration": asdict(config), "actual_device": device,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_prediction_positions": train_x.numel(), "development_prediction_positions": dev_x.numel(),
        "tokens_presented_to_loss": config.steps * train_x.numel(),
        "history": history, "empirical_bigram_train_loss_infimum": entropy,
        "training_loop_seconds": elapsed,
        "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "m_row_observed_successors": {tokenizer.tokens[j]: int(counts[m_id, j]) for j in range(tokenizer.vocab_size) if counts[m_id, j] > 0},
        "m_row_learned_probabilities": {tokenizer.tokens[j]: float(m_probabilities[j]) for j in range(tokenizer.vocab_size) if distribution[m_id, j] > 0},
        "corpus": "original toy sentences; no pretrained weights or TinyStories",
    }
    return model, tokenizer, result
