# StorySeed progress

Active fresh repository: Desktop/StorySeed, branch main, origin https://github.com/JackivGarg/StorySeed.git. Earlier folders remain references.

Phase 1 implementation, teaching material and verification are complete at the student's explicit request to deliver the whole phase. Understanding checks remain pending; no independent mastery is inferred.

| Subphase | Build / actual run / notes | Student evidence |
|---|---|---|
| 1.1: environment | Implemented, executed, chapter and solutions written | Pending |
| 1.2: tokens | Implemented, executed, chapter and solutions written | Pending |
| 1.3: batches | Implemented, executed, chapter and solutions written | Pending |
| 1.4: bigram | Implemented, executed, chapter and solutions written | Pending |
| 1.5: loss | Implemented, executed, chapter and solutions written | Pending |
| 1.6: training | Implemented, executed, chapter and solutions written | Pending |
| 1.7: generation | Implemented, executed, chapter and solutions written | Pending |
| 1.8: checkpoint | Implemented, executed, chapter and solutions written | Pending |

Environment: project venv shares the installed PyTorch 2.6.0+cu124 dependency. CPU and CUDA gradient examples actually ran. Full dependency isolation is an alternate documented setup, not the environment used here.

Phase 1 uses an authored toy corpus and a character bigram model. It is not the final transformer or TinyStories training.

Next implementation: Phase 2, after reviewing Phase 1 learning questions.

Phase 1 subphase commits through 5326d0c were successfully pushed to origin/main on 2026-10-14. The final study-summary commit follows separately. Git tracking state is authoritative for the latest publication status.

Actual results: 400 CUDA Adam updates, 289 parameters, 240 training/67 development prediction positions. Training loss 2.835582 to 1.212152; development loss 2.834067 to 1.216261. All eight contract tests passed. Saved weights and tokenizer reproduce the seeded result in a fresh process. CPU reproduction commands also ran. Local checkpoint: checkpoints/phase01/bigram.pt; weights remain ignored by Git, with a hash/manifest in reports/phase01/08_checkpoint.json.
