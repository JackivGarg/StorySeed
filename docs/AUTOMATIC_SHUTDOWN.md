# Automatic shutdown after the current training run

The user authorized power off after the unattended training run. A hidden local Python helper now watches the verified training PID and Windows process creation time. It uses Windows process waits; it does not require repeated AI messages or a Codex automation service.

Before requesting shutdown it requires the exact watched process to exit with code zero, the completed 48,000-update report, all 32,000 continuation updates, matching continuation protocol identity, and matching final full-checkpoint/inference/tokenizer hashes. The helper refused the current incomplete result during setup verification. It is armed and waiting, not a shutdown request already counting down.

After those conditions pass it requests Windows power off with a 120-second countdown. Save other open work: Windows closes applications at the end of the countdown. If the training process fails or completion validation fails, it records failed_without_shutdown and does not request shutdown. There is no promise of power off after a failed or interrupted run.

Keep the laptop plugged in and awake with the lid open, and keep Codex open for the existing native training session. Closing Codex may interrupt that training session. The shutdown helper is a separate hidden local process; it cannot make the existing training session durably tracked by Codex Process Jobs.

To cancel before training finishes, double-click cancel_automatic_shutdown.cmd in the project folder, or run `.venv/Scripts/python scripts/shutdown_after_training.py --cancel`. During the Windows countdown, `shutdown /a` also aborts the pending shutdown. Status is in runs/quality_recovery/auto_shutdown_status.json. A cancellation sentinel prevents a waiting helper from requesting shutdown later.

This shutdown covers the complete unattended training run and its saved artifacts. Final reserved-text evaluation, qualitative story review, notes closure, candidate promotion and application packaging remain pending for the next session. No improved final app or completed project is claimed by powering off.

The helper source is scripts/shutdown_after_training.py. Authorization/runtime logs remain in the ignored runs folder; the setup observation is reports/quality_recovery/AUTO_SHUTDOWN_SETUP.json. Authorization is consumed before issuing shutdown and cannot silently rearm on a later run. No permanent power-plan setting was changed.

Windows command behavior: [Microsoft shutdown documentation](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/shutdown).
