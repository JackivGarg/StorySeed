# StorySeed recovery4096 — model card

Educational English story-continuation model, explicitly implemented in PyTorch and trained from new random initialization. No pretrained language-model weights or hosted inference API are used. The original512-entry model remains a historical baseline; its artifacts are preserved.

## Architecture and training

8,473,088 parameters;8 pre-LayerNorm decoder blocks; width256;8causal heads of width32; GELU feed-forward multiplier4; residual paths; training dropout0.1; untied head; learned window-relative positions up to256 input IDs. Training-only byte BPE vocabulary4096 with PAD0/BOS1/EOS2. Explicit attention remains the implemented mechanism; no KV cache or retrieval augmentation.

Pinned TinyStories revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64. Source budget64 MiB,74,094 normalized-unique complete stories,58,968 training stories and13,138,578available targets. Pinned source card declares cdla-sharing-1.0; attribution Ronen Eldan and Yuanzhi Li. Raw stories are not distributed. This is a convenience source prefix, not a representative full benchmark. Exact normalized overlap checks do not establish semantic near-duplicate absence.

Training completed 48,000 AdamW updates with 150,743,877 scored-event presentations. The initial 16,000-update scratch run was followed by a separately declared 32,000-update full-state continuation because development diagnostics still confused story identities. No final text was accessed before that extension. Microbatch 16 × accumulation 2; up to 128 scored targets per 256-input row; BF16 eligible GPU operations, FP32 master weights; seed 31/data_seed 20261021; clipping 1, matrix weight decay 0.1. Initial LR: 200-update warmup to 0.0006, cosine to 0.00006. Continuation LR: new stage-local 200-update warmup to 0.00015, cosine to 0.00003. Weights, AdamW state and RNG were retained across the declared stage boundary. Protocols: planning/QUALITY_RECOVERY_PROTOCOL.json and planning/QUALITY_RECOVERY_CONTINUATION_PROTOCOL.json. Best fixed-development checkpoint: global update 48,000; NLL 1.730995 on 46,082 events. Gross recorded training elapsed: 6.26 hours including observed system pauses.

## Evaluation and limits

The selected model/tokenizer hashes were frozen before new final text encoding/scoring. On the same 200 stories/172,006 content bytes, original512scores1.707953 bits/byte and recovery4096scores0.626487. Token NLLs are not directly comparable across representations. Byte-normalized canonical encoded-path likelihood includes EOS, uses differing finite-context policies and is not a clean isolated ablation. One recovery training seed; paired story bootstrap varies stories conditional on fixed models.

Raw paired outputs and identified assistant review are included in reports/quality_recovery. Human ratings remain pending. The retained outputs show recurring reference confusion, repetition and weak event logic. Reliable story continuation has not been achieved. The release is an improved experimental prototype. This is not a general assistant or a production children's-content system; dataset naming does not establish child suitability. No publication acceptance or novel architecture is claimed.

Both original and new final populations are now used evidence. Further tuning needs a new declared evaluation design. The app uses fixed CPU FP32 weights and does not train during requests. Installed Python/dependencies remain required; inference is local after setup. Hashes support integrity/compatibility, not a signed publisher identity. See QUALITY_RECOVERY_REPORT.md and LOCAL_APP.md.
