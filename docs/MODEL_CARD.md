# StorySeed main128 — model card

## Identity and purpose

StorySeed is an educational English short-story continuation model. Its decoder and training pipeline were explicitly implemented in PyTorch; main weights were trained from random initialization. PyTorch supplies tensor operations/autograd, and the tokenizers library supplies the saved BPE backend. No pretrained language-model weights or hosted inference API are used.

Primary use: learning language-model engineering and demonstrating a bounded university experiment. It is a learning prototype, not a reliable general assistant or a production children's-content system. Generated text may be malformed, incoherent or inappropriate; the dataset name does not establish suitability for children.

The released application retains the predeclared main128 artifact. The shorter-context study condition scored better in one paired experiment; the app does not quietly replace the main model after examining final scores.

## Architecture

- 6,605,312 parameters; decoder-only, eight pre-LayerNorm blocks.
- Feature width 256, eight causal attention heads, head width 32.
- Feed-forward expansion 4 with GELU; residual paths and dropout 0.1 in training.
- 512-entry byte-level BPE, including PAD/BOS/EOS IDs 0/1/2.
- Learned window-relative positions, maximum 128 supplied IDs; untied vocabulary head.
- Explicit attention and masks; no KV cache, retrieval component or instruction/chat fine-tuning.

## Data and training

Source: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), Eldan and Li, pinned revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. The pinned dataset card declares `cdla-sharing-1.0`; see the included corpus manifest and upstream terms/attribution. The project does not assert that this dataset declaration automatically defines a separate license for model weights or project source. Raw stories are not distributed in this release.

The experiment used an 8 MiB convenience source prefix: 9,562 normalized-unique complete stories, of which 7,585 were assigned to training by the unchanged normalized-group hash rule. Training pool: 2,818,569 available next-ID targets. Original development: 105 stories/39,711 targets. Final: 120 stories/44,282 targets, released only after fixed completed artifacts; this final population has now been used.

The tokenizer was fit only on the earlier training split and retained unchanged. Main training: seed 17, 2,000 successful AdamW updates, microbatch 4 rows × accumulation 4, up to 64 scored target IDs per row, 1,884,373 true-target presentations, sampling with replacement. LR peak 0.0003, warmup 100, cosine to 0.00003, gradient norm clip 1; CUDA BF16 eligible operations with FP32 master weights. This is less than one pool's worth of target presentations and does not imply full epoch coverage. Full settings and sampling/provenance hashes are in planning/PHASE07_08_PROTOCOL.json and the Phase 7 reports.

## Observed results

| Model | Final mean NLL (nats/ID) | Perplexity |
|---|---:|---:|
| Main128 | 2.777847 | 16.084 |
| Context64 sibling | 2.609649 | 13.594 |
| Expanded-data count bigram, alpha 0.1 | 3.201461 | 24.568 |
| Historical learned bigram | 3.265621 | 26.196 |

All score the same 44,282 target events. Main full-development NLL changed from 6.295087 to 2.805321. The context pair matched parameter count, initial state, actual training row sequence and true-target/update budgets, not compute. One paired training seed was used. Main-minus-short final NLL difference is +0.168199; story-bootstrap 95% interval [+0.155971,+0.180609] describes story variation conditional on these fixed models, not uncertainty across training seeds.

The operational stall/recovery and raw timing are preserved. Separate matched resource observations do not establish an intrinsic speed ranking. See reports/phase08/PAPER_DRAFT.md and CONTEXT_STUDY.md for controls, limitations and actual resource measurements.

## Qualitative limitations

The frozen packet contains 24 continuations: six prompts × two sampling seeds × two conditions. Identified assistant review found familiar openings/dialogue fragments alongside grammar errors, malformed words, character drift and weak prompt continuity. Low within-output repetition did not establish coherence. Independent human ratings remain pending; no human preference score or reviewer agreement is claimed.

Limited data exposure, token fragmentation and position/context policy are possible explanations, not individually isolated causes. The model is not instruction-tuned, cannot reliably answer questions, and forgets inputs outside its current window. Future quality claims require a newly declared protocol and fresh untouched evaluation population rather than reuse of the final set for tuning.

## Local application and artifacts

CPU FP32; one request at a time; default seed 17, temperature 0.8, top-k 40, at most 48 new IDs. PAD/BOS sampling forbidden, EOS allowed. Controls cap new IDs at 256 and prompts at 4,000 characters. Full visible prompt can exceed the 128-ID supplied context. Inference does not update parameters or read raw training stories.

Main checkpoint SHA-256: `f0f7241af4770372738c3c9b057e205cc60e66cc7fa1d21397a6281f383c0eed`.
Tokenizer SHA-256: `3540aba1c244ed063b9fb317f0d62280a63d00725cac980e16c3b583cdf549cb`.

The local release includes weights/tokenizer/runtime source and manifests, not Python or dependency binaries. No remote prompt transmission or automatic prompt logging is implemented; explicit Save result writes JSON locally. Dependency installation may require internet. Artifact hashes support integrity/compatibility; the manifest is unsigned. See LOCAL_APP.md.

## Completion status

Implementation, recorded training/evaluation and authored teaching are separate from learning attainment. Independent student reconstruction, independent human ratings, other-machine dependency installation and publication acceptance are not claimed. This is a documented university project, not a novel architecture claim.
