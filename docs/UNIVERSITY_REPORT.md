# StorySeed: Building and Evaluating a Small Transformer for Story Continuation

**Team:** StorySeed Lab
**Domain:** Natural Language Processing and Deep Learning
**Project type:** Implemented educational system with an exploratory controlled experiment

## Abstract

StorySeed implements a decoder-only language model in PyTorch and trains its main weights from random initialization to continue short English stories. The project connects story-level data preparation, byte-level BPE, explicit causal attention, complete transformer blocks, a recoverable training workflow and local text generation. A bounded TinyStories prefix supports a 6.6-million-parameter model within laptop GPU resources. A fixed-budget study compares 128-input and 64-input context policies while keeping architecture and sampled target/update budgets matched. On 44,282 fixed final next-ID events, the main model obtains negative log-likelihood 2.777847 and the shorter-context condition 2.609649. Both outperform the reported bigram references, although generated continuations retain grammar and continuity errors. A local interface and portable inference bundle demonstrate the trained system. The contribution is transparent implementation and measured analysis under explicit constraints; no new transformer algorithm or independent human preference result is claimed.

## 1. Problem domain and motivation

Text generation requires predicting plausible continuations from an opening. A one-token bigram can learn local transitions but cannot condition directly on a broader prefix. Larger hosted models can produce fluent text, but using an API alone does not expose the model-building and training decisions needed for this learning objective. StorySeed therefore focuses on constructing and understanding a small causal model under a measured resource budget.

The intended user can enter a short story opening, generate a continuation, adjust decoding controls and save the output with its settings. The model is a continuation prototype; it is not a general chatbot or a dependable children's-story author.

## 2. Existing approaches and project position

| Approach | Capability | Relevant limitation or distinction |
|---|---|---|
| Count/learned bigram | Simple local token transitions | Direct conditioning uses one current ID |
| Fixed-window neural predictor | Learns shared representations and ordered context | Finite rigid input structure; our short pilot did not beat its baseline |
| Causal transformer | Context-dependent retrieval and layered feature processing | More data, optimization and compute are needed to use its capacity |
| Pretrained or hosted language model | Stronger broad language generation | Different objective from explicitly learning scratch training |

StorySeed uses established transformer methods. Its useful distinction is the complete inspectable build, shared-tokenizer references, controlled context-policy comparison, preserved execution evidence and local deployment. It is not claimed to outperform modern pretrained models. A project can earn technical merit through clear implementation and valid analysis without inventing novelty.

## 3. Objectives

1. Build the causal decoder, data path and training workflow explicitly in PyTorch.
2. Train a feasible small model from random weights on documented story data.
3. Compare declared context policies using compatible target units and matched controls.
4. Interpret numerical and qualitative outcomes, including negative results and limitations.
5. Provide a runnable local application and a guided route to independent reconstruction.

## 4. Methodology and architecture

The data pipeline captures a revision-pinned 8 MiB TinyStories source prefix, normalizes duplicate groups and applies the unchanged content-based split rule. It retains 7,585 training stories, the original 105 development stories and the original 120 final stories. The original training-only 512-entry byte BPE tokenizer is retained unchanged. Training batches preserve story boundaries and distinguish scored targets from padding and warm-up context.

The decoder has 6,605,312 parameters, eight pre-LayerNorm blocks, width 256, eight causal heads, feed-forward multiplier 4, GELU, residual paths, training dropout 0.1 and learned window-relative positions. The output head predicts 512 logits. Cross-entropy measures next-ID prediction error; autograd and AdamW optimize the random-initialized parameters. Main training uses microbatch 4, accumulation 4, CUDA BF16 eligible operations, FP32 master weights, LR warmup/cosine decay and clipping. Full-state checkpoints preserve optimizer, RNG and data state for recovery.

```mermaid
flowchart LR
    A[Pinned stories] --> B[Grouped splits]
    B --> C[Saved BPE and masked batches]
    C --> D[Random-initialized causal decoder]
    D --> E[Cross-entropy and AdamW updates]
    E --> F[Fixed inference artifacts]
    F --> G[Declared final evaluation]
    F --> H[Local CLI and desktop app]
```

Both context conditions complete 2,000 successful updates with identical initial parameter state, sampled row sequence and 1,884,373 true-target presentations. Input width differs (128/64), while the architecture, including the 128-entry position table, remains fixed. This isolates an implemented context policy under matched target budgets; it does not match compute or remove all position/dropout consequences of differing input shapes.

## 5. Relevance of techniques

BPE gives a finite vocabulary with byte coverage. Shared embeddings turn IDs into learned features. Causal attention permits use of earlier inputs without future-target leakage. Multiple heads and feed-forward transformations support different feature interactions. Residual paths and LayerNorm support repeated transformations. Cross-entropy connects correct-target probabilities to an optimization objective. Accumulation provides the intended effective batch within memory limits. Checkpointing makes the long run recoverable rather than merely restartable. Temperature and top-k control sampled continuations without changing learned parameters.

## 6. Evaluation and observed findings

| System | Final NLL | Perplexity | Targets |
|---|---:|---:|---:|
| Main128 | 2.777847 | 16.084 | 44,282 |
| Context64 | 2.609649 | 13.594 | 44,282 |
| Expanded count bigram | 3.201461 | 24.568 | 44,282 |
| Historical learned bigram | 3.265621 | 26.196 | 44,282 |

Main full-development NLL falls from 6.295087 to 2.805321 across the declared training run. That supports improved prediction on that fixed development population; it is not an accuracy percentage or proof of coherent stories. The shorter-context condition has lower final NLL in this one paired seed. The story-bootstrap interval is conditional on the fixed trained models and does not measure variation across independent training runs. The historical baseline used less data and a different budget; the count baseline fits all expanded transitions through a different procedure. Neither is an architecture-only matched-compute comparison.

The final population was scored only after the full artifacts were fixed; it has now been used. An identified assistant inspected all 12 paired prompt/seed comparisons and found substantial grammatical and narrative limitations in both conditions. Independent human ratings remain blank/pending. No human satisfaction score is invented.

A system/device stall affected the raw context64 duration. Its cause was not established. An execution-only allowance amendment restored the same saved training process without changing the scientific update/target budget. Raw timing is retained; a separate matched resource observation is reported in the paper draft, without claiming an intrinsic speed ranking.

## 7. Feasibility and working application

The model was trained on the available RTX 3050 Laptop GPU with 6 GB VRAM after a resource pilot. The delivered application runs CPU FP32 using the exact main checkpoint/tokenizer. It provides CLI and Tkinter desktop input, decoding controls, generated output, copy and JSON save. A shared service keeps inference behavior consistent; worker computation leaves the UI event loop responsive.

A local release packages runtime source, trained weights, tokenizer and provenance. It requires compatible Python/dependencies; it is not a standalone executable or an offline dependency installer. An outside-repository fresh-process observation records source relocation and output identity on this machine. Other-machine setup is not asserted without observation.

## 8. Scope, limitations and future work

The source prefix is a convenience subset, not a representative benchmark. One paired training seed cannot establish universal context superiority. Bounded exposure, a small vocabulary and finite context constrain the learned behavior; their individual causal contributions were not isolated. Independent human ratings, repeated training seeds and broader quality evaluation remain future work. Any further tuning must distinguish the already-used final data from a new untouched evaluation population.

Future research can predeclare multiple-seed repeats and targeted ablations, evaluate them on new reserved data, and measure narrative coherence with identified reviewers. More training, a different tokenizer or a new architecture component should each be motivated and measured rather than assumed to improve quality.

## 9. Conclusion

StorySeed demonstrates the full path from story data to a scratch-trained small transformer and a local application. Its strongest outcomes are explicit implementation, traceable recovery, compatible measurements and honest interpretation of a negative context result. The model learned better next-ID predictions while retaining visible story-generation limitations. Student mastery is a separate outcome to demonstrate through reconstruction and explanation.

## References and project evidence

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Eldan and Li, [TinyStories](https://arxiv.org/abs/2305.07759), 2023; pinned dataset provenance in data/manifests/phase07/corpus.json.
- [PyTorch model saving/loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
- Local evidence: reports/phase07/SUMMARY.md; reports/phase08/metrics.json, CONTEXT_STUDY.md, ASSISTANT_SAMPLE_REVIEW.md and PAPER_DRAFT.md; reports/phase09 application records.

This is a university project report, not a claim of publication, acceptance or a new state-of-the-art result.
