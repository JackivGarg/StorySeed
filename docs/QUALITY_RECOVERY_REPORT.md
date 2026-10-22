# StorySeed quality recovery: from failed continuations to a measured new model

## Why the extra work was necessary

The original project delivered a functioning scratch-training and application pipeline, but the user's blue-fox example exposed unusable grammar and poor prompt continuity. Engineering completion was not a sufficient acceptance criterion for a story-continuation model. The recovery preserves that failure and introduces a separate declared design, learning record and evaluation population.

## Diagnosis

Bounded sampled-row observations found correct next-ID shift, scored masks and story boundaries. Earlier logits did not change when future IDs were altered in the causal check. Padding and final-projection differences were below1e-4. The screenshot prompt used21old IDs including BOS, so initial cropping was not the explanation. These observations rule against the checked mechanisms, not every possible bug.

The original6.6M model received only1,884,373scored target presentations, about0.669of its available-target pool, sampled with replacement. It also used512-entry BPE,128input IDs and a48-new-ID application cap. Insufficient learning exposure was a strong suspect; fragmentation/context and output length were additional measured limitations, not individually isolated causes.

## Combined recovery design

Expand the same pinned source to 64 MiB, fit 4,096-entry BPE on training only, increase context to 256 IDs, and train a new 8,473,088-parameter explicit decoder from random initialization. Declare an initial 16,000-update budget with recoverable state. Its diagnostic samples still confused entities, so a separate 32,000-update continuation and new stage-local schedule were declared before any fresh-final access. This is a development-informed extension, not a claim that 48,000 updates were the original unchanged plan. The expanded train contains 58,968 stories. New development/final groups exclude all 9,562 groups from the old 8 MiB prefix. Each new reserved population has 200 stories. Exact normalized group overlap is zero; semantic duplicates are not ruled out.

On the same new development text, the old tokenizer uses1.694times as many content IDs. None of those200stories fit BOS plus complete content within the old window;152fit under the new representation/window. This measures available context capacity, not successful learned narrative memory.

## Actual training and selection

The measured resource pilot used about 1.1 GB allocated GPU memory. Training completed 48,000 updates/150,743,877 scored-event presentations in 6.26 gross recorded hours on the RTX 3050 Laptop GPU. Full-state checkpoints were saved every 250 updates. The first stage continued in the same native process across recorded system sleep interruptions; EXECUTION_INCIDENT.json preserves those observations. The second stage deliberately loaded the completed first stage's full state; 03b_STAGE_BOUNDARY.json records equality of actual parameters, AdamW state, scaler, random states and row generator at that transition. The gross duration includes pauses and is not GPU compute time. Windows rejected Codex Process Jobs before launch, so native finite execution was used, not an unsupported tracked background job.

Initial fixed-development NLL8.342617fell to the selected value1.730995at update48,000. Selection used only the fixed-development criterion. Early samples documented readable words before consistent narrative behavior; those intermediate failures remain preserved.

## Same-text final measurement

After freezing the selected artifact, original/recovery models scored the same200freshly reserved stories/172,006bytes. Canonical encoded-path bits/byte were1.707953and0.626487, respectively (63.3%lower for recovery). This is a compatible normalization of differently segmented text, not a direct comparison of raw token NLL or a percentage story-quality improvement.

The complete paired packet includes32raw outputs under frozen prompts/seeds/settings. Identified assistant observations are in ASSISTANT_REVIEW.md. Independent human ratings remain pending. All samples, not just a favorable story, must inform the quality description.

## Application changes

The prepared application identifies the reviewed model/tokenizer pair, reports loaded context/parameter metadata, offers the historical model through --original, raises the default new-ID budget to 192 and shows generated continuation separately from the prompt. CLI and Tkinter retain the same sampler/service. Runtime source, trained tensors, tokenizer and evidence are packaged locally; Python/dependency binaries are installed separately. CPU/Tk and fresh-process package observations are recorded separately in 05_app.json and 06_portable_app.json once available. They assess integration rather than certify story quality.

## What the experiment does and does not establish

This is an end-to-end engineering recovery: data, tokenizer, context, initialization and training budget change together. It does not isolate each ingredient's causal effect or establish multi-seed superiority. Better prediction/word formation does not guarantee every continuation is coherent. Both final populations are now used evidence; new tuning requires separately reserved evaluation. Student reconstruction, independent human assessment and publication acceptance are not claimed.

The contribution is a traceable diagnosis, resource-aware scratch retraining, compatible comparison and reproducible local deployment. Established transformer techniques remain credited; no new architecture is invented.
