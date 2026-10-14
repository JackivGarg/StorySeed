# Learn from each natural commit

The fresh history begins at a332e7a. Do not import earlier repository history or fabricate separate past subphase commits. Local commits are authorized by the user's build-and-teach request. Use the existing configured identity, stage task-owned files and preserve unrelated changes.

Each subphase normally contains the focused implementation, detailed lesson files, small observations and index/state updates. The body names subphase, reason, reading order, code, actual evidence and next increment. The subject describes the change naturally:

```text
feat: implement a trainable bigram language model

Subphase: 1.4
Read: notes/phase01/04-bigram-scores.md
Code: src/storyseed/models/bigram.py
Evidence: reports/phase01/04-observations.md
Learning: introduced; student checkpoint pending.
Next: probabilities and cross-entropy.
```

Paths above are examples of future files, not files already created. Review the diff and relevant meaningful checks before committing. Avoid test suites that merely mirror the implementation; verify important properties such as alignment, causality, accumulation and resuming where applicable.

To study a commit:

```powershell
git log --oneline
git show --stat <commit>
git show <commit>:notes/phase01/04-bigram-scores.md
git show <commit> -- src/storyseed/models/bigram.py
```

Read the lesson at the same revision as the diff. Do not overwrite uncommitted work by checking out history just to read it. Use git show, or an isolated worktree for historical execution when needed.

Long runs have a configuration/lesson commit before execution and a later outcome commit. Checkpoints/corpora stay outside Git; small manifests identify artifacts and hashes. A recorded launch is not a completed training result.

Record prior hashes during a later change: a commit cannot contain its own final hash. Final handoffs give the actual hash, lesson links, observed behavior, a few supportive checks and next increment.

Remote origin is configured and authentication/pushing succeeded. Phase 1 subphase history was published through 5326d0c. Check actual Git tracking state for the latest synchronization and update PROGRESS.md when publication changes. Never force-push or expose credentials to make history appear complete.
