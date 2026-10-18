# Phase 9 — make the trained system usable and reconstructible

Base: 4ae59ab. Five increments follow the existing end-to-end plan. Main128 remains the predeclared main artifact, not a new selection based on the used final population.

1. Consistent CPU generation service/CLI: validate settings, retain the exact main checkpoint/tokenizer identities and existing sampling semantics.
2. Tkinter local desktop interface: load once, run inference off the event loop, show/save/copy real output and report failures. No remote service or new GUI dependency.
3. Portable source-and-weight bundle: package the exact checkpoint, tokenizer, runtime source, provenance, dependency pins and launchers. Observe a fresh process from outside the repository. This is not a bundled Python interpreter or an offline dependency installer.
4. Model card, university report and short live-demo narrative using actual measured evidence and clearly pending human ratings.
5. Connected reconstruction workbook and substantial separate phase09 concept chapters. Written guidance is complete only when authored; student demonstration remains pending until actually performed.

Execute local inference and an actual UI demonstration to collect application evidence. No automated test suite, new training, hyperparameter search or final-set rescoring is planned. Preserve phase01–08 chapters and existing measured results. Keep weights/archive/output JSON local and ignored. Natural commits record each focused increment; the final synchronization records actual publication.

The application defaults to CPU FP32, one thread, seed 17, temperature 0.8, top-k 40, maximum 48 new IDs, 128 supplied context slots. Controls bound prompts to 4,000 characters and new IDs to 256. Empty prompts use BOS. Long prompts retain their full displayed text while inference uses only the rightmost context IDs. These are usability limits, not newly measured model-quality claims.
