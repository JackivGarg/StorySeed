# Phase 1 final contract checks

Executed `.venv/Scripts/python -m unittest discover -s tests -v`.

Observed: eight tests passed in 0.569 seconds of test execution. Python/package startup is outside that number.

The checks cover document boundaries, the final valid window, a hand-derived selected-row gradient, stopping on EOS without printing the marker, seeded sampling and mode restoration, rejected input/settings, exact checkpoint state/token/sample restoration, and rejected malformed/nonfinite checkpoint tensors.

The checkpoint lesson separately launched scripts/generate.py in a fresh process and compared its complete JSON response with the original model's configured response. That child exited 0 and matched exactly.

The recorded CLI training/generation outputs in 08_cli_training.json and 08_cli_generation.json exercise the public reproduction commands on CPU, using a separate ignored checkpoint so the primary CUDA artifact stays unchanged. These files contain actual output, not expected fixtures.

These checks establish the Phase 1 mechanics. They do not establish coherent story quality, future transformer behavior, complete optimizer resuming or student understanding.
