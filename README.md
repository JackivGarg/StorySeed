# StorySeed 🌱

**A small language model trained from scratch for short English story continuation.**

StorySeed takes a story opening and generates what comes next, one token at a time. The project brings together a custom Transformer, text tokenization, model training, evaluation, and a local desktop application.

The model was trained from random initialization on a subset of **TinyStories**, using PyTorch. Its predictions come from learned weights, with inference running locally.

## Project highlights

- **GPT-style Transformer:** explicitly implemented causal self-attention, positional embeddings, feed-forward layers, residual connections, and layer normalization.
- **Custom text pipeline:** byte-level BPE tokenization, story-level data splits, and next-token training examples.
- **GPU training:** mixed precision, gradient accumulation, and checkpoints that preserve training state.
- **Local application:** a desktop interface with adjustable sampling settings, story copying, and saved generation results.
- **Recorded evaluation:** learning curves, comparisons with the original model, and retained story outputs.

## Model details

| Detail | Value |
|---|---|
| Parameters | 8.47 million |
| Transformer layers | 8 |
| Attention heads per layer | 8 |
| Vocabulary size | 4,096 tokens |
| Context window | 256 tokens |
| Training stories | 58,968 |
| Development / final-test stories | 200 / 200 separate stories |
| Training updates | 48,000 |

## From an opening to a continuation

```text
Story opening → tokenization → embeddings → Transformer
             → next-token prediction → generated continuation
```

The model repeatedly predicts and selects the next token using the recent story context. The application offers seed, temperature, top-k, and output-length controls to explore different continuations.

## Results

The original and improved models were evaluated on the same **200 final-test stories**.

| Model | Predictive loss: bits per UTF-8 content byte ↓ |
|---|---:|
| Original StorySeed | 1.708 |
| Improved StorySeed | 0.626 |

The improved model achieved **63.3% lower predictive loss per byte**. This measures prediction performance, rather than story-quality accuracy. The comparison combines changes to data, tokenization, context, and training budget.

![Original and improved model comparison](docs/media/quality-recovery/03-same-text-comparison.png)

## Limitations

Generated text is more readable than the original model's output, but it can still repeat, confuse characters, change objects, or produce inconsistent events. StorySeed is an educational prototype, and reliable narrative coherence is not established. The recorded qualitative review was performed by an identified assistant; independent human quality ratings are pending.

## Technology

**Python · PyTorch · NumPy · Hugging Face Tokenizers · Tkinter**

## Project evidence

- [Model card](docs/RECOVERY_MODEL_CARD.md)
- [Project report](docs/QUALITY_RECOVERY_REPORT.md)
- [Generated story review](reports/quality_recovery/ASSISTANT_REVIEW.md)
- [Charts and exact statistics](docs/media/quality-recovery/)

## Acknowledgments

Training text comes from **TinyStories**, introduced by Ronen Eldan and Yuanzhi Li. Dataset attribution and source details are recorded in the model card.
