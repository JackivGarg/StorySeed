# Practical teaching rewrite — 2026-10-14

The user requested replacing textbook/exam notes with guided engineering sessions. Updated both project and notes-folder instructions, teaching standards, the phase plan and reading maps. Rewrote eight Phase 1 sessions and eight hands-on labs; added a guide tracing the complete implemented data flow.

Canonical notes: C:/Users/JACKIV/OneDrive/Desktop/StorySeed-Study-Notes.
Original archive: history/059c13b-before-practical-rewrite in that folder.
Model implementation reference before the rewrite: 059c13b.

## Checks completed during this rewrite

- Verified 18 original note files in the archive against their historical Git contents, allowing only Git/Windows newline representation.
- Checked Markdown links in all 22 current note/instruction/index files: no broken local targets.
- Parsed all 30 Python snippets: no syntax errors.
- Executed the first complete standalone inspection in each of the eight lab files using the project interpreter from the code repository root. All eight completed, including their assertions.
- Confirmed the saved main checkpoint's SHA-256 stayed unchanged after those inspections.

The lab inspections used existing functions. The generation lab trained a separate tiny CPU score table in memory and did not save it. Historical main-run training reports and trained checkpoint remain the source of the previously reported CUDA results. The full existing unittest suite was not rerun for an editorial change.

## Useful observed lab result

The generation lab's temperatures 0.7, 1.0 and 1.5 all produced “mia sat m t.” with seed 17 in the inspected CPU run. This is consistent with changed distributions yielding identical sampled IDs for some seeds. The same-last-token prefix comparison returned “mia m ca m t.” and “sam m ca m t.” with identical generated ID sequences.

## Revision record

planning/LESSON_MANIFEST.json records exact current external file hashes. It detects changed note content but does not publish or store those notes in the code repository. Preserve external history for future revisions. Student understanding remains pending; successful assistant inspections are not mastery evidence.
