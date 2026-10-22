# Identified assistant review of all retained continuation pairs

Reviewer: Codex assistant, 2026-10-22. This is an identified, nonblind qualitative review, not independent human evidence or a scored human study. All 32 full raw continuations were read, comprising eight declared prompts, two seeds and two fixed models. The review did not use only the 80-word display excerpts. Model identity: 5abebfe480e5f50fddba62cd9227d22b434a69248fc2c647cc5e403d7b2805e0.

## What the actual packet supports

In all 16 inspected pairs, recovery word formation and sentence structure are substantially clearer than the original's fragmented text. The new model sometimes develops a recognizable event and supplies an ending. The corrected fox opening at seed 17 is one of the stronger examples: a second fox named Sammy is introduced and they play together.

Reliable story continuation is not established. Recurrent failures include confused character references, unexplained object transitions, repetition, weak causal connections and failure to develop the opening's actual problem. The exact typo-containing fox example still conflates fox/butterfly abilities. A grammatically readable paragraph can therefore remain a poor story.

The deployment decision is to replace the broken local prototype with the measurably better experimental prototype, retaining the original. This is an improvement in basic language modeling and readability; it is not acceptance of the broader goal of consistently coherent stories. No additional training budget is silently started. A next intervention requires a separately reasoned design, not an assumption that more updates guarantee a fix.

## Pair-by-pair observations

Prompt numbers are one-based here; raw JSON uses zero-based prompt_index. Original strings, settings, generated IDs and stopping reasons remain unchanged in 04_samples.json.

| Prompt | Seed | Original512 observation | Recovery4096 observation |
|---:|---:|---|---|
| 1 | 17 | Original invents togree/Suife and loses grammatical structure. | Lily/tree exploration has readable dialogue; have reach is malformed, the flight/voice is weakly grounded and the passage stops mid-sentence. |
| 1 | 29 | Original contains facers, compe and disconnected events. | Timmy helps a bird and reaches EOS, but the broken wing becomes a broken rock and a stick supposedly heals it; the final moral has a pronoun error. |
| 2 | 17 | Exact typo opening produces terffy, notict and unrelated people. | Fox/butterfly journey reaches an ending with intact words; the characters' flying abilities and distress are confused without a clear causal transition. |
| 2 | 29 | Original has broken clauses and invented words. | Fox/rabbit interaction is readable and reaches EOS, but pinching is abruptly framed as happiness and the final clause is ungrammatical. |
| 3 | 17 | Corrected fox opening still produces terffy and unrelated names. | A second fox is introduced, named Sammy, and the pair play/explore before The end. This is one of the stronger continuations; speaker attribution is ambiguous and dialogue repeats. |
| 3 | 29 | Original loses the fox and contains many malformed words. | Fox/mouse/tree topic persists, but the mouse swims up a tree, safety motivation shifts and the mouse and the mouse appears with repeated safe. |
| 4 | 17 | Original loses Mia and the box in fragmented dialogue. | Starts by trying to open the box, then repeatedly has a bird leave a box/room with unstable locations and duplicated shoes; reaches the ID limit. |
| 4 | 29 | Original switches to unrelated names and invented actions. | Mia/box/key/doll are recognizable; putting the doll on the doll and repeated doll references destroy clear actions. It ends unfinished. |
| 5 | 17 | Original cannot maintain a grammatical account of helping. | Recognizable Tom/friend/toys conflict, but it says both he did not want to share and he wanted to share; helping is replaced by unresolved hostility. |
| 5 | 29 | Original contains malformed actions and shifts pronouns. | A paper-cutting activity develops with a friend and reaches EOS; cutting scissors/paper, who thanks Tom, and the sudden firefighter ambition remain confused. |
| 6 | 17 | Original drifts among a girl, bird and unrelated objects. | Bird/balloon play remains readable and reaches EOS, but the original fear of rain is abandoned and balloon/bird agency is sometimes unclear. |
| 6 | 29 | Original uses broken clauses and invented words. | Bird remains the subject but attempts to climb the ground and grab wind; fear repeats and there is an unusual booked against his feathers phrase. Length-stopped. |
| 7 | 17 | Original changes Sara to Lily/Timmy and loses the toy search. | Sara begins searching but repeats closet and rocks, then switches to a rocket/superhero scene without resolving the missing toy; length-stopped. |
| 7 | 29 | Original is lexically fragmented and changes characters. | Sara/book/words passages are readable but abandon the toy search, repeat words and end with an unsupported self-judgment; length-stopped. |
| 8 | 17 | Original loses the rabbit amid invented words and people. | Rabbit/light/button topic has a visible ending; unlocking a button/light and celebrating a failed opening do not form a clear causal story. |
| 8 | 29 | Original has fragmented language and unrelated references. | Rabbit/girl meeting reaches EOS, but holding a hand and a kiss is odd, the girl becomes the rabbit and repeated goodbyes muddy the ending. |

## Compare lengths and stopping honestly

Observed stops: {"original512": {"length": 16}, "recovery4096": {"eos": 9, "length": 7}}. The recovery has nine EOS stops and seven length stops; all original outputs hit the length cap. Both use at most 192 new IDs, temperature 0.8 and top-k 40. Different tokenizers make that different word budgets; an EOS stop is not a coherence rating. Character and causal failures above were assessed from actual text, not inferred from stopping statistics.

The prompts are authored diagnostic tasks, not an independent random sample of all user requests. The two seeds are sampling seeds for fixed weights, not retrained model seeds. GPU FP32 generation uses the recorded historical full-padding layout; the app's compact CPU execution can choose different IDs because small numerical changes can cross sampling boundaries. Actual CPU output is checked separately.

## Connect to the fixed final likelihood evidence

The candidate was frozen at global update 48,000 before new final text encoding/scoring. Both models scored the same 200 stories / 172,006 UTF-8 content bytes. Canonical encoded-path bits per content byte are 1.707953 for original512 and 0.626487 for recovery4096. EOS losses are included, and differing segmentation/context remain visible confounds. The 63.3% reduction is not a percentage improvement in story quality. This is a combined data/tokenizer/context/training recovery with one recovery training seed.

The paired story-bootstrap interval describes story variation conditional on the fixed artifacts, not training-seed variation. Both the original and new final populations are now used evidence and cannot remain fresh test sets for further tuning. Human ratings, child suitability and student mastery are not established by this review.
