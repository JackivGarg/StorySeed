# 1.7: the model supplies its own next input

During training, the observed text supplied the next-token answers. Generation changes that situation: we have an unfinished prompt and must select a continuation ourselves. We now implement that loop explicitly in [generation.py](../../src/storyseed/generation.py). [07_generation.py](../../lessons/phase01/07_generation.py) trains the same tiny configuration, compares untrained/trained samples and records [actual outputs](../../reports/phase01/07_generation.md). Saved-model loading comes in the next subphase.

## Connect next-token prediction to a sequence

Suppose the prompt is `mia `. Encode it without the ending marker. Its last token is a space, so the bigram selects the space row of W. Softmax gives probabilities for the next character choices. We select one token, append it to the growing sequence, then ask again using the newly selected last token.

This is autoregressive generation: previous outputs become subsequent inputs. A useful mental calculation is a two-choice distribution after a space, for example `[0.75 for m,0.25 for s]`. Sampling can select either. If it chooses `m`, the next step uses the `m` row, perhaps selecting `i`. The result depends on both learned distributions and random selections.

This illustrative distribution explains the mechanism; our actual probabilities come from trained weights. We do not write story fragments into the generator or retrieve an original sentence. The resulting text can nonetheless resemble training phrases because the learned transitions came from them.

## Why generation differs from training

The training input at each position is an observed corpus token. The generated input is a previously chosen output, which might lead to combinations not present in the corpus. The model has no access to an already-written future continuation. Repeated small plausible choices do not guarantee globally sensible text.

Our model is especially limited: a different prompt history with the same final character gives the same starting distribution. The demonstration compares `mia m` and `sam m` under the same sampling seed. They produce identical new token-ID sequences, proving that their earlier characters do not influence the continuation. A transformer will eventually use a permitted prefix instead of this single-token state.

## Sampling is a choice, not another training update

`torch.multinomial` draws a categorical choice from the probability vector. The name can sound complicated, but we request one draw: one token index. A dedicated CPU random generator provides a recorded seed. Recreating the generator and using the same model/configuration reproduces the sample in our supported setup; it does not prove deterministic training across all hardware.

We put generation under `torch.no_grad()` and evaluation mode, restoring the prior model mode afterward. Inference does not update parameters. Sampling is performed on CPU from copied score values so an identical CPU-loaded score table can reproduce a sample from the same seed. This extra transfer is acceptable for our tiny experiment, not an optimized serving strategy for a large model.

The forward calculation has shape `[1,1,V]`: one example, one current token, V scores. Selecting `[0,0]` gives the vector `[V]`. The growing text is stored as token IDs, but this bigram never processes that whole list as contextual features.

## Temperature and top-k

For temperature τ > 0, we scale logits as `z_j/τ` before softmax. For two logits `[ln(3),0]`, τ = 1 gives probabilities `[0.75,0.25]`. At τ = 0.5, exponentials are `[9,1]`, giving `[0.9,0.1]`. At τ = 2, they are `[sqrt(3),1]`, giving approximately `[0.634,0.366]`.

Lower temperature concentrates choices; higher temperature spreads them. Neither necessarily improves story quality. Temperature zero is rejected because division by zero does not implement greedy decoding.

Top-k keeps only the k highest scores and renormalizes after assigning excluded choices negative infinity. Our implementation selects exact returned indices, so ties cannot accidentally retain more than k choices. With k = 1, the highest-scoring token is chosen at each step. On a bigram this can produce a cycle, such as repeatedly returning to a common character sequence, rather than a story. The script records an actual bounded k=1 example rather than assuming it is more coherent.

## Stopping and supported prompts

Generation stops when it selects `<eos>` or exhausts `max_new_tokens`. The token budget counts sampled tokens, including an ending marker if one is sampled. The readable text hides that marker. A zero budget returns the original prompt with no draws.

An empty prompt is rejected because this model has no learned beginning-of-story input distribution. Unknown characters are rejected by the strict tokenizer, and invalid temperature/top-k values produce useful errors. This is a limited educational vocabulary, not arbitrary-English input support.

## Interpret the actual outputs

Compare the report's before/after text and repeat-seed checks. The untrained table supplies nearly uniform choices; the trained table supplies learned transition patterns. In this run the same prompt and seed changed from `mia tcrsdwnsahmeddmt` before training to `mia sat m t.` afterward. The k=1 sample is `mia sat.`, while another seed produced `mia on.`. The more readable example does not erase the broken ones. These are local transition patterns, not evidence of consistent story generation.

Next we preserve trained weights and tokenizer identity, then reproduce generation in another process without retraining. Before that, use [the practice](07-practice-and-solutions.md) to explain precisely which parts of this loop are learned and which are application decisions.

Reference: [PyTorch categorical sampling operation](https://pytorch.org/docs/stable/generated/torch.multinomial.html). The temperature calculations above are hand-worked examples, not benchmark results.
