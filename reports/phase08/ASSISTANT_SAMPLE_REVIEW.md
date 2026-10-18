# Identified assistant review of all paired continuations

Reviewer: Codex assistant. Date: 18 October 2026. Evidence: all 24 outputs in `samples.json`, twelve prompt/seed pairs under the frozen settings. This is qualitative assistant analysis, not independent human grading. No human scores, inter-rater agreement or human preference percentages are supplied; `review_ratings.csv` remains pending.

## What the samples actually show

Both models have learned recognizable story openings, short clause patterns, common names and dialogue punctuation. Both still produce malformed words, unstable references and actions that fail to develop the prompt. The 64-slot model's lower held-out NLL therefore does not establish fluent, coherent story generation.

The following observations cover every paired task rather than selecting a flattering example. “Long” means main128 and “short” means context64; these are model identities, not sample lengths.

| Prompt / seed | Long continuation | Short continuation | Supported reading |
|---|---|---|---|
| Story opening / 17 | Introduces bird Sara, then unexplained Suife and malformed words such as “togree” and “parble.” | Also introduces Sara and a new-friend goal; later has “sunglain,” “treab” and an unclear group transition. | Both show learned opening structure; the short opening is locally easier to follow, but neither sustains a sensible complete story. |
| Story opening / 29 | Lily's play narrative breaks into “facers,” “compe” and awkward references. | Lily gives way to unexplained Timmy and malformed actions. | Shared story-template competence coexists with grammar and character-continuity failures. |
| Mia/red box / 17 | Moves to excitement and mother dialogue without developing the red box. | Introduces Sarah/Lily and dialogue, also without developing the box. | Neither adequately follows the prompt's object or event; low exact repetition would miss this failure. |
| Mia/red box / 29 | Adds Larah and Tima with invented words and unclear action. | Starts with an unexplained “He,” switches to green toys and ends with malformed phrasing, despite emitting EOS. | An end marker is not proof that the box narrative was resolved. |
| Tom/helping friend / 17 | Introduces Jack and nonsensical actions instead of a clear helping event. | Switches among references, mother, dog and Jack without resolving the helping goal. | Both lose the intended goal; punctuation and quoted speech are insufficient evidence of coherent interaction. |
| Tom/helping friend / 29 | Generic play/park passage shifts to unrelated objects and a mother. | Generic outdoor play and a fox do not establish how Tom helped his friend; it ends with EOS. | Mechanically finishing and fulfilling the prompt are separate observations. |
| Bird/rain / 17 | Switches to a little girl and unrelated actions rather than addressing fear of rain. | Mentions the bird being happy, then adds Timmy and malformed dialogue; EOS occurs. | The short sample retains the bird briefly but does not explain a coherent rain-related resolution. |
| Bird/rain / 29 | Reuses an outdoor/park template and ends with unrelated Joe dialogue. | Switches from bird context to boy/girl references and invented phrases. | Prompt-specific conditioning is weak in both; an existing fluent fragment does not repair the overall continuation. |
| Sara/missing toy / 17 | Adds books, Lily and invented actions without a clear search. | Adds books, an old man and mother dialogue without finding the toy; EOS occurs. | Neither resolves the missing-toy setup. |
| Sara/missing toy / 29 | Reuses the park template with malformed objects and dialogue. | Introduces a fox and Ben with broken phrasing and punctuation. | Both drift from Sara's problem; changing character names is not story development. |
| Rabbit/bright light / 17 | Becomes a boy/Joe/Jack dialogue with fractured wording and no explanation of the light. | Adds friend/mother/Jack fragments and ends with a replacement glyph. | Word composition and continuity are still imperfect; neither develops the inciting event. |
| Rabbit/bright light / 29 | Reuses outdoor/park wording and changes objects/references. | Uses outdoor/forest/toy fragments and unstable pronouns without returning to the light. | Both favor generic learned patterns over the supplied event. |

## Connect these observations to the metrics

Main128 has mean duplicate word-trigram fraction **0.0** across these twelve continuations, yet the review above shows substantial quality problems. The metric checks exact repeated three-word sequences within an output. It does not check grammar, plot, reference consistency, invented words or reuse of a generic opening across different prompts.

For example, “liked to play outside inside in the park” recurs across several seed-29 main128 continuations. That cross-output template reuse is visible even when each individual continuation has no duplicate trigram. This is a qualitative observation made after reading the retained packet, not a newly predeclared primary metric or a claim that all outputs are identical.

Context64's mean duplicate-trigram fraction is approximately **0.001323**, also very low. Its mean unique-word fraction is approximately 0.7145 versus 0.6745 for main128, but generated length differs and malformed words can increase apparent diversity. Those ratios do not establish superior coherence.

Main128 reaches the 128-new-ID cap in all twelve samples. Context64 emits EOS in four of twelve; its mean length is about 122.67 IDs. The paired examples show why EOS termination must not be described as successful narrative completion.

## What the held-out result supports

Context64 NLL is 2.609649 versus main128's 2.777847 on the same 44,282 final targets. That supports better observed next-ID prediction by the short policy in this fixed one-seed experiment. It does not override the visible story failures or establish that shorter context is generally better.

Limited exposure, small-vocabulary fragmentation and how context/positions are supplied are plausible directions for further investigation. This experiment does not isolate their individual causes. Additional optimization or new design choices would be a separately declared follow-up; the now-used final population should not become hidden tuning data.

## What remains pending

An identified human can now read the complete A/B packet without first opening the identity key, fill the grammar/continuity/consistency/repetition rubric and record comments. Until actual ratings exist, no human review averages or agreement results belong in the university paper. Learner understanding also remains pending until the student can explain and modify the system independently.
