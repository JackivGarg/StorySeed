# Reconstruct StorySeed through your own decisions

This is a guided route to demonstrate understanding, not a record that the student has already rebuilt the model. Use a separate scratch directory or a new branch/worktree when ready; preserve the current trained artifacts and used-final evidence. Do not silently retune against the used final population.

For each milestone, write your reasoning before consulting the finished source. Use the actual implementation afterwards to locate differences. Keep a personal evidence log: what you predicted, what you implemented, what you observed, what failed and how you diagnosed it. Ordinary API documentation is allowed; memorizing entire files is not the objective.

## 1. Recover the task from a story

Choose one authored sentence, not a new hidden evaluation case. State how BOS/content/EOS form observed next-ID events. Show why input/target sequences are shifted and why EOS is a real target. Decide how story boundaries and padding should work before writing a dataset function.

Produce a small hand-worked input/target/mask example. A reconstruction is convincing when you can explain why the last content ID predicts EOS, why PAD never counts as an answer and why concatenating independent stories without a boundary policy changes the task.

Hint: compare with src/storyseed/story_batches.py after reasoning. Verify by decoding and counting target events, not by trusting tensor dimensions alone.

## 2. Recover representation and context

Explain why integer IDs need shared learned feature vectors. Choose V, D and maximum T; derive token-table `[V,D]` and position-table `[T,D]` shapes and parameter counts. Explain what is shared across occurrences, what receives gradients and why equal vocabulary size does not establish equal token meaning.

Build a small numerical illustration of lookup plus position addition. Distinguish position count from vocabulary index. Explain why our positions count real tokens within the supplied window and why PAD must not invent earlier linguistic content.

Hint: consult token_features.py and position_features.py after the explanation. Verification: changing a real token should change its lookup features; adding only left padding must not advance the real position numbers.

## 3. Recover causal retrieval

Start with the need: the current token must retrieve relevant earlier information. Derive Q/K/V roles, `[B,H,T,d]` projected-head shapes, `[B,H,T,T]` score shape, scaling by sqrt(d), allowed-source masks, row-wise softmax and weighted-value retrieval. Explain why the softmax axis represents candidate sources.

Construct a tiny three-position causal mask on paper. Then implement one head before multiple heads. Explain why a PAD-only query row needs a safe all-masked treatment rather than softmax of only negative infinities.

Hint: attention.py and decoder_components.py support comparison. Verification: a future token must not influence an earlier valid output; valid attention mass should be distributed only over allowed sources. State which assertion corresponds to which conceptual rule.

## 4. Recover the full decoder

Explain why retrieval alone is not the whole block. Derive positionwise D→4D→D transformation, GELU, residual paths, pre-LayerNorm ordering, dropout behavior and final normalization/head. Keep `[B,T,D]` consistent across residual additions. Choose an architecture small enough to inspect before considering the 6.6M model.

Draw the actual implemented block, not a remembered post-normalized diagram. State whether the output head shares weights with the token table (ours is untied). Count parameter groups and reconcile with decoder.parameter_ledger.

Verification: explain how training-mode dropout differs from eval and how inference mode differs from both. Preserve the mathematical role of each component before copying the exact API syntax.

## 5. Recover learning

Derive masked summed cross-entropy divided by true-target count. Explain the `p-one_hot` logit derivative, how autograd reaches shared parameters, why gradients must be cleared and where clipping/optimizer/schedule happen. Construct two unequal-sized microbatches and derive the intended accumulation normalization.

Use a tiny fixed training batch only to inspect whether learning can memorize known events. Then create a separately described development measurement. Do not present tiny memorization as generalization. A full experiment must separate data used for updates, data used for decisions and final evidence.

Hint: loss.py, training_runner.py and training_optimizer.py. If loss fails to fall, inspect example alignment, loss mask/denominator, finite logits, gradient presence, optimizer parameter membership and update movement before adding layers.

## 6. Recover state and experimental meaning

Explain why an inference export needs architecture, weights and tokenizer, while a faithful training resume additionally needs optimizer, schedule/progress, random state and data sampling state. Explain how a runtime interruption differs from a planned algorithm change. State which context-study controls matched and which did not.

Use actual Phase 7–8 reports to explain NLL, perplexity, paired story bootstrap and the limits of one seed. Write a paragraph interpreting the shorter-context win without claiming universal superiority. Explain why the final set can no longer be called untouched after evaluation.

Verification: classify a proposed change as inference policy, new training, operational recovery or a new experiment. Decide what evidence each change would require.

## 7. Recover inference and the application

Implement the autoregressive loop from your own sketch: encode/BOS, crop/pad context, forward last-position logits, temperature, forbid PAD/BOS, top-k, seeded sample, EOS/length stop, decode. Explain why no optimizer or raw training corpus belongs in this path. Describe what a long displayed prompt does and does not imply about model memory.

Then design the service/request boundary and interface lifecycle. Explain why background computation sends data through a queue while the main thread owns widgets. Finally list the files that must travel in a portable release and separate installed dependencies from bundled artifacts.

Hint: context_generation.py, app_service.py, desktop.py and release_bundle.py. Verification: with the fixed release/runtime/settings, compare saved generated IDs. A matching sample is useful functional evidence, not proof that every implementation detail is correct or that stories are good.

## Evidence log template

| Milestone | My prediction/design | My implementation | Observed behavior | Diagnosis/next step |
|---|---|---|---|---|
| Fill after doing the work | Pending | Pending | Pending | Pending |

The mentor can review this log and your reconstruction later. Until then, implementation/notes are delivered and student demonstration remains pending. You are aiming to explain and rebuild the chain of decisions, not claim expertise simply because the finished repository exists.
