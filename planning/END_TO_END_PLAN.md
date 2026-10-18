# StorySeed: fresh build, detailed teaching and natural commits

## Current completion — Phase 9

All nine planned build phases have delivered implementation and detailed separate teaching. Phase 9 adds the local fixed-main app, source/weight/tokenizer release, model card, university report and reconstruction workbook. Read reports/phase09/SUMMARY.md for scoped actual execution. Student demonstration and independent human ratings remain pending. No new training or final scoring occurred; earlier entries are historical. The final complete release is identified by reports/phase09/FINAL_RELEASE.json after packaging closure.

## Historical progression

Active location: `C:/Users/JACKIV/OneDrive/Desktop/StorySeed`.
Repository: https://github.com/JackivGarg/StorySeed .
Created from a332e7a with no imported source or history. This plan carries the agreed learning sequence forward as new project guidance, not a claim that earlier code exists here.

## Outcome and scope

Build a compact causal transformer in PyTorch and train its main weights from random initialization for simple English story continuation. PyTorch supplies tensors, gradients and device execution; we explicitly implement the model's architecture, data path, training and generation. TinyStories is the candidate public corpus. Record its version, subset, terms and attribution before use.

The student's outcome is independent reconstruction and explanation using detailed notes and ordinary API documentation. The application, trained checkpoint and university report support that learning goal. Main model size is provisionally around 6–12M parameters, subject to an actual pilot. Previous hardware observations suggest an RTX 3050 Laptop GPU with 6 GB VRAM and about 16 GB RAM; recheck locally before planning training.

## Phase/subphase sequence

There are 49 core subphases. Phase 1's eight build increments, runs and teaching files are now complete; student understanding checkpoints remain pending. Phase 2 is now implemented and taught on its documented teaching pilot. Phase 3 is implemented on the same pilot; Phase 4 is implemented on the same teaching pilot; Phases 5 and 6 are implemented and taught with bounded pilots; Phases 7 and 8 are implemented and executed; Phase 9 is implemented with separate concept-first teaching; independent student reconstruction remains pending. Each row describes a coherent build-and-teach sequence, with run-specific decisions recorded in the corresponding reports.

| Phase | Subphases in order | Working outcome |
|---|---|---|
| 1: first learning loop | 1.1 environment/tensors/gradients; 1.2 character vocabulary and shifted examples; 1.3 tensor windows/batches; 1.4 trainable bigram scores; 1.5 softmax/cross-entropy; 1.6 updates and training; 1.7 sampling/generation; 1.8 save/reload | A tiny trained model with an understood one-token context limitation |
| 2: story pipeline | 2.1 inspect/document TinyStories subset; 2.2 separate story splits/check overlap; 2.3 small worked BPE implementation; 2.4 train/serialize main tokenizer; 2.5 efficient boundary-aware batches; 2.6 train common-tokenizer baseline | Reproducible data, tokenizer and fair baseline |
| 3: learned representations | 3.1 embedding lookup; 3.2 fixed-window neural predictor; 3.3 short training/comparison; 3.4 context limitations and useful common interfaces | A contextual ANN baseline that motivates attention |
| 4: causal attention | 4.1 Q/K/V projections; 4.2 scaled scores/softmax/weighted values; 4.3 causal mask; 4.4 positions; 4.5 integrated attention model | A worked and runnable attention mechanism |
| 5: complete decoder | 5.1 multiple heads; 5.2 feed-forward sublayer; 5.3 residuals/LayerNorm; 5.4 dropout/train-eval modes; 5.5 stack blocks and count parameters; 5.6 tiny-batch learning/causality checks | A complete explicitly implemented GPT-style decoder |
| 6: dependable training | 6.1 configuration and development evaluation; 6.2 optimizer/schedule; 6.3 correct gradient accumulation; 6.4 measured precision options; 6.5 full checkpoint/resume; 6.6 memory/throughput pilot | A repeatable training system and feasible budget |
| 7: main training | 7.1 freeze run protocol; 7.2 run/record training; 7.3 inspect curves and generation failures; 7.4 reload/export inference bundle | Scratch-trained weights with traceable evidence |
| 8: evaluation/research | 8.1 freeze evaluation rubric/protocol; 8.2 prediction/repetition/resource metrics; 8.3 reviewed paired story samples; 8.4 controlled context study; 8.5 results analysis and paper draft | Evidence-backed comparisons and honest limitations |
| 9: application and independence | 9.1 CLI generation; 9.2 simple local interface; 9.3 reproducible packaging; 9.4 university report/model card/viva; 9.5 guided independent reconstruction | Runnable system and demonstrated understanding |

Optional later work: one justified modern component, or a separately documented pretrained-model adaptation project. Neither replaces scratch-trained StorySeed.

## A subphase is both a change and a lesson

Teach the concept and decision journey: identify a missing project capability, reason about an approach, learn its prerequisite in depth, return explicitly to the original build question, implement the motivated component, interpret the result and discover the next limitation. Source-file order does not determine the learning order. The user approved this style on 2026-10-15. All current Phase 1 teaching is rewritten in that direction; use it for later phases as they are implemented. Keep the 49-subphase implementation sequence, actual code and existing observations as references. Follow the 2026-10-15 teaching rules.

Write the detailed chapters in the separate Desktop StorySeed-Study-Notes folder. Commit the implementation, small observations and updated lesson-map/state together. Commit messages describe the actual change, for example `feat: implement a trainable bigram language model`. The commit body states subphase, external lesson paths, code paths, evidence and next dependency. A job-start/configuration commit and a later result commit accurately represent long runs.

## Planned file organization

`C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes/phaseXX/`: substantial concept chapters, examples, code walkthroughs and practice/solutions, outside the code repository.
`lessons/phaseXX/`: bounded, inspectable learning experiments.
`src/storyseed/`: reusable data, model, training and generation components.
`configs/` and `scripts/`: reproducible settings and commands.
`reports/phaseXX/`: small observations and interpreted results.
`data/manifests/`: source/split/tokenizer provenance.
`planning/`: sequence, rules, learning evidence and experiment protocol.

Introduce directories when needed. Raw/processed data, large checkpoints, runs, caches and environments are local artifacts outside Git. Notes live separately on the Desktop; planning/LESSON_MAP.md associates them with the code subphases. Existing notes in older Git commits remain available without rewriting history. The previous Desktop study folders remain historical references.

## Research controls and realistic budget

Candidate question: how context length affects story quality and resource cost. Candidate contexts 64/128/256 depend on the pilot. Train conditions separately with a common tokenizer/splits, fixed architecture including maximum position-table size, matched target-token budgets and documented optimizer/data-sampling controls. Use comparable held-out target positions and defined prefixes. Equal token counts do not imply equal compute; measure both. Repeat seeds when feasible and report the actual count.

Loss/perplexity, repetition, human-scored grammar/continuity and resource measurements support different claims. A fluent example does not prove quality, and established methods do not become novel algorithms through implementation. Keep final cases out of model selection. Publication acceptance requires the actual university/venue rules and cannot be promised.

Provisional learning allowance: 180–260 active hours, revised after Phase 1 and the GPU pilot. Research repeats and optional extensions may add time. No model quality, training runtime or mastery is assumed.

## Current start

Phase 1 is implemented in this fresh repository with eight natural subphase commits, actual execution reports, GPU training and fresh-process checkpoint reproduction. Its eight external teaching sessions and eight lab files were rewritten on 2026-10-14 in the user's implementation-first mentor style, with a phase guide and a complete data-flow guide. Original notes remain archived externally. The model is an authored-corpus character bigram with 289 trainable parameters. It is not the final transformer. The project venv explicitly shares the already-installed PyTorch dependency; a fully isolated installation is a documented alternate path. Next: work through the Phase 1 sessions and then Phase 2's documented story corpus. Earlier folders remain references rather than imported source/history.

Primary references for later teaching: https://docs.pytorch.org/tutorials/beginner/basics/ , https://arxiv.org/abs/1706.03762 , https://arxiv.org/abs/2305.07759 , https://huggingface.co/datasets/roneneldan/TinyStories . Explain methods originally and credit any implementation reference actually used.

## Approved teaching coverage — 2026-10-15

The external Phase 1 guide, eight core chapters, eight reasoning companions, full-system reconstruction, environment companion and coverage map follow the approved concept-first flow. THE_SLM_JOURNEY.md explains why each later phase is needed. Later detailed chapters remain implementation-dependent; this documentation revision does not implement Phases 2–9 or establish learner mastery.

## Phase 2 completion — 2026-10-16

The six story-pipeline increments were implemented and executed with their conceptual chapters/companions in external phase02. Source prefix: 1 MiB, 1,129 complete stories. Splits: 904 training, 105 development, 120 reserved holdout. Main tokenizer: 512 byte-level BPE entries, training-only fit. Batches preserve each target once with padding masks. A random-initialized compatible bigram ran 200 count-weighted Adam updates and was reloaded in a fresh process. The final holdout remains unencoded/unscored; learner understanding is pending. Next architecture increment: Phase 3 learned representations and contextual ANN.

## Phase 3 completion — 2026-10-17

Four subphases implemented and executed with separate external phase03 chapters: learned token features, ordered finite-context ANN, fixed-budget K=1/8 training, and common inference with boundary observations. Training setup and completed outcomes have separate commits. Both neural conditions learn but do not beat the Phase 2 bigram under this budget. The unchanged tokenizer/splits/target positions anchor the comparison; capacity/compute are not matched. Fresh-process inference reproduces the saved sample. See reports/phase03/SUMMARY.md. Next: Phase 4 causal attention; final holdout remains unencoded/unscored and student evidence pending.

## Phase 4 completion — 2026-10-17

Five subphases are written, implemented and executed: Q/K/V projections, scaled retrieval, causal/padding restrictions, learned position features and an integrated single-head predictor. Training setup and outcome are separate commits. The 300-update pilot learns but development NLL 4.653214 does not beat prior baselines. Whole-model information/gradient observations, a common evaluator and fresh-process generation are recorded. See reports/phase04/SUMMARY.md and external phase04/README.md. Next: Phase 5 complete decoder. Earlier chapters and final-holdout reservation are preserved; learner understanding remains pending.

## Phase 5–6 completion — 2026-10-18

Phases 5 and 6 each have six natural implementation commits and separate Desktop phase05/phase06 concept chapters, companions, reconstructions and LOSS_AND_LEARNING.md. Read reports/phase05/SUMMARY.md and reports/phase06/SUMMARY.md before Phase 7. Tiny-batch NLL 6.242738 → 0.001790 is memorization of 124 repeated events. The separate ten-update CPU pilot reduces full development NLL 6.246756 → 5.843923 on 39,711 targets. CPU fresh-process full-state resume matches its recorded reference. The 6,605,312-parameter CUDA candidate is a resource pilot; main training remains Phase 7. Preserve earlier Phase 1–4 chapters and the final holdout. Learner understanding remains pending.

The user explicitly requires teaching how and why loss changes in relation to what we actually did. Carry this into every future phase: derive the metric, connect gradients/updates and data exposure to its movement, state population/mode/timing/mask denominator, use actual charts where useful, explain generalization and confounds, and diagnose realistic alternatives. Separate observed changes from hypotheses. Never invent intermediate curves or claim an individual component caused a combined improvement without a controlled measurement.

## Completed main experiment and fixed evaluation — 2026-10-18

Phases 7–8 source, actual execution and separate concept-first notes are complete, with identified assistant qualitative review and a university paper draft. Independent human ratings and learner understanding remain pending. Main128/context64 each completed 2,000 random-initialized training updates on the pinned expanded corpus (7,585 stories / 2,818,569 available training targets). Main development NLL is 6.295087 → 2.805321. Read reports/phase07/SUMMARY.md, reports/phase08/SUMMARY.md and CONTEXT_STUDY.md for actual final scores and limits. The main inference export reproduces in a fresh public CLI.

The original 120 final stories were encoded/scored only after both complete last-update artifacts and matched controls were frozen. This is now a used final population; do not tune on it and call it untouched. Earlier Phase 1–6 chapters remain byte-preserved historical evidence. There is one paired training seed; the paired story-bootstrap interval does not estimate training-seed uncertainty. Human forms are blank/pending; assistant review is not independent human evidence. Phase 9, local application/packaging and independent reconstruction, remains next.

Codex Process Jobs rejected Windows before a successful job launch. The finite workload used direct session 9576, saved the stalled comparison at update 669, and completed through resumed session 92051 with full checkpoint support. The raw timing incident and execution-only allowance amendment are preserved; separate matched resource observations avoid treating the stalled gross duration as a clean speed comparison. No background completion notification or unsupported CPJ job ID is claimed. Continue teaching actual loss changes, decisions and limitations with the same concept-first flow.
