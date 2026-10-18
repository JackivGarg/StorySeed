# A clear five-minute StorySeed demonstration

## Before presenting

Use the existing local weights or the final portable release. Open the app before the presentation so a missing dependency is found during preparation. Keep reports/phase09/01_generation.json as a labeled recorded fallback if a live runtime fails; do not pretend a fallback is live output.

From the active project: `.venv/Scripts/python -m storyseed.app --gui`.
From the extracted local release: `.venv/Scripts/python launch.py --gui` after setup.

## 0:00–0:45 — explain the problem

“We built a small decoder-only transformer to continue English story openings. We implemented the architecture in PyTorch and trained its weights from random initialization. Our goal was to understand the complete pipeline and study a controlled context-policy question on available hardware.”

Do not say that we implemented CUDA/autograd from scratch or trained a ChatGPT-equivalent model.

## 0:45–1:30 — show the system

Describe the path: documented stories → grouped splits → saved BPE → masked targets → causal decoder → cross-entropy/AdamW → saved inference export → local app. Main dimensions: 512 vocabulary, width 256, eight heads, eight blocks, 128-input context, about 6.6M parameters.

## 1:30–2:30 — generate once

Use the default opening/settings and click Continue story. Explain that generation keeps weights fixed and samples repeatedly from next-ID scores. Show the stop reason and token budget. Save result if useful. Read the actual continuation honestly: familiar opening structure does not hide malformed words or loss of continuity.

Changing seed or temperature explores a decoding policy; it does not retrain or prove better quality. Do not repeatedly generate until a good sample appears and present only that as representative.

## 2:30–3:30 — show the result table

Main final NLL 2.777847, context64 2.609649, expanded count bigram 3.201461, historical learned bigram 3.265621, all on 44,282 final targets. Lower is better for this metric; smaller context wins in this one paired experiment. Compute and historical baseline training budgets differ. The final data is now used evidence.

## 3:30–4:15 — explain one real engineering decision

Choose one you understand: why target counts exclude padding/warmup, why tokenizer bytes must match, or why full-state resume includes optimizer/RNG/sampler state. The operational interruption was recovered without changing the planned successful-update budget; gross duration was not silently used as a clean speed benchmark.

## 4:15–5:00 — limitations and next research step

“This is a learning prototype. Independent human ratings and multi-seed repeats remain pending. We would predeclare the next question and reserve new evaluation data before further tuning.”

## Questions you should be able to explain in your own words

- **From scratch?** Random-initialized language-model parameters, explicit decoder/data/training/generation, using PyTorch tensors/autograd and a library BPE backend. No pretrained LM weights.
- **Why transformer?** It lets positions retrieve relevant earlier information with causal masks; more capacity is not a guarantee under a fixed short budget.
- **What research contribution?** A transparent bounded context-policy study and reproducible analysis, not a new architecture. A paper can report controlled findings/limitations; acceptance depends on venue standards and further evidence.
- **Does lower loss mean good stories?** It means higher assigned probabilities to observed next IDs on that population. Narrative quality needs additional evidence; our samples still contain errors.
- **What did you personally do?** Describe the decisions/code you can actually explain and your use of AI assistance accurately. Do not claim independent reconstruction or solo authorship that you have not demonstrated; follow your institution's disclosure requirements.

This guide supports explanation after study. It does not record that the student has already mastered these answers.
