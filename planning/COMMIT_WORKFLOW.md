# Learn from each natural commit

The fresh history begins at a332e7a. Do not import earlier repository history or fabricate separate past subphase commits. Local commits are authorized by the user's build-and-teach request. Use the existing configured identity, stage task-owned files and preserve unrelated changes.

Each subphase commit contains the focused implementation, small observations and lesson-map/state updates. Detailed lessons are written alongside that work in the separate Desktop StorySeed-Study-Notes folder; they are not committed to the code repository. The body names subphase, reason, reading order, code, actual evidence and next increment. The subject describes the change naturally:

```text
feat: implement a trainable bigram language model

Subphase: 1.4
Read: Desktop/StorySeed-Study-Notes/phase01/04-bigram-scores.md
Code: src/storyseed/bigram.py
Evidence: reports/phase01/04_bigram.md
Learning: introduced; student checkpoint pending.
Next: probabilities and cross-entropy.
```

The paths above identify delivered Phase 1 files. Review the diff and relevant meaningful checks before committing. Avoid test suites that merely mirror the implementation; verify important properties such as alignment, causality, accumulation and resuming where applicable.

To study a commit:

```powershell
git log --oneline
git show --stat <commit>
git show <commit>:planning/LESSON_MAP.md
git show <commit> -- src/storyseed/bigram.py
```

Use the chapter map for commits after the notes relocation. Open the corresponding lesson in the separate Desktop folder. Preserve prior lesson versions there before content revisions, labelled with their code commit. External lesson files are not automatically versioned by this Git repository. For the original Phase 1 commits before relocation, use `git show <commit>:notes/phase01/04-bigram-scores.md` to read the original historical lesson. Do not overwrite uncommitted work by checking out history just to read it. Use git show, or an isolated worktree for historical execution when needed.

Long runs have a configuration/lesson commit before execution and a later outcome commit. Checkpoints/corpora stay outside Git; small manifests identify artifacts and hashes. A recorded launch is not a completed training result.

Record prior hashes during a later change: a commit cannot contain its own final hash. Final handoffs give the actual hash, lesson links, observed behavior, a few supportive checks and next increment.

Remote origin is configured and authentication/pushing succeeded. Phase 1 subphase history was published through 5326d0c. Check actual Git tracking state for the latest synchronization and update PROGRESS.md when publication changes. Never force-push or expose credentials to make history appear complete.
