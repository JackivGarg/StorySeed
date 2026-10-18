# Existing loss observations for the teaching charts

The JSON companion copies the ten recorded public-CLI training observations and two full-development observations. It records the local run report hash. No new training/evaluation was launched for these charts.

Training losses are measured before their associated updates on changing sampled rows with dropout. Development measurements use the same full population at initialization and after ten successful updates. Intermediate development points were not observed and are not invented.

Read the separate Desktop phase05/LOSS_AND_LEARNING.md and phase06/LOSS_AND_LEARNING.md for the connected interpretation. The plotting script `scripts/plot_learning_records.py` generates artifacts outside the code repository.
