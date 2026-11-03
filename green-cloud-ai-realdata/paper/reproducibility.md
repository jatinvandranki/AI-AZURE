## Reproducibility

Run `python run_experiment.py --mode dataset-check` followed by `python run_experiment.py --mode full`. The configured seeds are [101, 202, 303, 404, 505]. Raw Azure CSV files are read from a relative data directory and are excluded from git. Reward configuration is selected on the validation split only; the test split is evaluated after selection.
