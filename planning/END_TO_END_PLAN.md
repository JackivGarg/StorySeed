# StorySeed: fresh build, detailed teaching and natural commits

Active location: `C:/Users/JACKIV/OneDrive/Desktop/StorySeed`.
Repository: https://github.com/JackivGarg/StorySeed .
Created from a332e7a with no imported source or history. This plan carries the agreed learning sequence forward as new project guidance, not a claim that earlier code exists here.

## Outcome and scope

Build a compact causal transformer in PyTorch and train its main weights from random initialization for simple English story continuation. PyTorch supplies tensors, gradients and device execution; we explicitly implement the model's architecture, data path, training and generation. TinyStories is the candidate public corpus. Record its version, subset, terms and attribution before use.

The student's outcome is independent reconstruction and explanation using detailed notes and ordinary API documentation. The application, trained checkpoint and university report support that learning goal. Main model size is provisionally around 6–12M parameters, subject to an actual pilot. Previous hardware observations suggest an RTX 3050 Laptop GPU with 6 GB VRAM and about 16 GB RAM; recheck locally before planning training.

## Phase/subphase sequence

There are 49 core subphases. Phase 1's eight build increments, runs and teaching files are now complete; student understanding checkpoints remain pending. Phases 2–9 are planned. Each row describes a coherent build-and-teach sequence, with run-specific decisions recorded in the corresponding reports.

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

Start from the previous working state and its limitation. Revise relevant prior knowledge. Explain the next concept with intuition, worked mathematics and defined shapes. Implement it, inspect actual behavior, teach the interpretation and give supportive checkpoints. Follow TEACHING_RULES.md and NOTES_STANDARD.md.

Normally commit the implementation, detailed chapter files, small observations and updated index/state together. Commit messages describe the actual change, for example `feat: implement a trainable bigram language model`. The commit body states subphase, lesson paths, code paths, evidence and next dependency. A job-start/configuration commit and a later result commit accurately represent long runs.

## Planned file organization

`notes/phaseXX/`: substantial concept chapters, examples, code walkthroughs and practice/solutions.
`lessons/phaseXX/`: bounded, inspectable learning experiments.
`src/storyseed/`: reusable data, model, training and generation components.
`configs/` and `scripts/`: reproducible settings and commands.
`reports/phaseXX/`: small observations and interpreted results.
`data/manifests/`: source/split/tokenizer provenance.
`planning/`: sequence, rules, learning evidence and experiment protocol.

Introduce directories when needed. Raw/processed data, large checkpoints, runs, caches and environments are local artifacts outside Git. Notes stay versioned beside code. The previous Desktop study folders remain historical references.

## Research controls and realistic budget

Candidate question: how context length affects story quality and resource cost. Candidate contexts 64/128/256 depend on the pilot. Train conditions separately with a common tokenizer/splits, fixed architecture including maximum position-table size, matched target-token budgets and documented optimizer/data-sampling controls. Use comparable held-out target positions and defined prefixes. Equal token counts do not imply equal compute; measure both. Repeat seeds when feasible and report the actual count.

Loss/perplexity, repetition, human-scored grammar/continuity and resource measurements support different claims. A fluent example does not prove quality, and established methods do not become novel algorithms through implementation. Keep final cases out of model selection. Publication acceptance requires the actual university/venue rules and cannot be promised.

Provisional learning allowance: 180–260 active hours, revised after Phase 1 and the GPU pilot. Research repeats and optional extensions may add time. No model quality, training runtime or mastery is assumed.

## Current start

Phase 1 is implemented in this fresh repository with eight natural subphase commits, sixteen concept/practice Markdown files, actual execution reports, GPU training and fresh-process checkpoint reproduction. The model is an authored-corpus character bigram with 289 trainable parameters. It is not the final transformer. The project venv explicitly shares the already-installed PyTorch dependency; a fully isolated installation is a documented alternate path. Next: review Phase 1 learning questions, then Phase 2's documented story corpus. Earlier folders remain references rather than imported source/history.

Primary references for later teaching: https://docs.pytorch.org/tutorials/beginner/basics/ , https://arxiv.org/abs/1706.03762 , https://arxiv.org/abs/2305.07759 , https://huggingface.co/datasets/roneneldan/TinyStories . Explain methods originally and credit any implementation reference actually used.
