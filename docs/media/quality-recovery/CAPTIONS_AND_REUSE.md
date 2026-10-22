# StorySeed charts — captions and reuse guide

These are copies of the existing measured charts, not recreated or edited pictures. PNG files are ready to upload. SVG files preserve vector detail for slides, reports and resizing. The original study-note assets remain intact. STATS_AND_CHARTS.json records the actual values, source revision and file hashes.

## Current recovery images

### 01-full-recovery-learning

Fixed-development NLL decreases from 8.342617 to 1.730995 over 48,000 successful updates, using the same 4,096-entry representation and the same 200 development stories / 46,082 targets. The dashed boundary at 16,000 marks a separately declared continuation with a new learning-rate schedule. Faint training points use changing sampled batches and training dropout. The curve is not comparable directly with old-tokenizer per-ID NLL.

### 02-continuation-detail

The additional 32,000 updates reduce fixed-development NLL from 1.862129 to 1.730995, about 7.0% lower. The vertical axis is deliberately cropped and labeled. The initial boundary regression and subsequent gradual improvement are actual observations. This is not a 7% story-quality score or a controlled isolation of training duration.

### 03-same-text-comparison

After model selection/freeze, both models score the same 200 newly reserved final stories / 172,006 UTF-8 content bytes. Canonical encoded-path bits per content byte decrease from 1.707953 to 0.626487, a 63.3% reduction. EOS losses are included. Different tokenization/context/data/compute remain confounds. This is not a 63.3% improvement in story coherence, grammar or accuracy.

### 04-initial-16000-learning

The original recovery budget reduces fixed-development NLL from 8.342617 to 1.862129. Its diagnostic stories still confused entities, prompting the separate continuation. This historical first-stage chart stops at 16,000; it is not the final recovery curve.

## Reusable short description

StorySeed is an educational GPT-style language model implemented explicitly in PyTorch and trained from random initialization. The recovery uses 58,968 training stories, an 8.47M-parameter decoder and 48,000 updates across two declared stages. Prediction and readability improved substantially; recurring character-reference and plot errors remain. Independent human ratings are pending.

## Historical images

The historical folder preserves earlier phase charts under their original phase paths. Their populations, tokenizers and budgets differ from the current recovery. In particular, tiny-batch memorization curves do not demonstrate unseen-story quality. Preserve original titles and refer to the corresponding phase reports when using them.

## Before sharing

Use the current-recovery images for the current model. State the measurement population and units in the caption, keep the cropped-axis disclosure for the continuation detail, and retain the quality limitations. All 16 retained prompt/seed pairs were reviewed by the identified assistant; this is not independent human evidence. Both final populations are now used evidence. No architecture novelty or publication acceptance is claimed.

Saved for later reuse; no LinkedIn post has been published.
