# Sustainable Green Cloud Computing Using Self-Optimizing AI Agents

Trace-driven research code for evaluating a self-optimizing two-agent AI system on Microsoft Azure Public Dataset V2 CPU traces.

The project estimates energy with a transparent server power model. It does not claim measured Azure data-center electricity consumption.

## Run

```bash
pip install -r requirements.txt
python run_experiment.py --mode quick
python run_experiment.py --mode standard
python run_experiment.py --mode dataset-check
python run_experiment.py --mode full
```

`quick` is for debugging, `standard` is for development, and `full` is the only mode intended for paper results. Full mode validates that enough intervals exist and fails loudly when the local trace sample is too small.

## Dataset

Place Azure V2 CPU shards and the VM table under `data/azure_v2`, or pass `--data-dir`. Supported trace files are `.csv` and `.csv.gz`. Raw dataset files are ignored by git.

For the current publication target, run:

```bash
python setup_azure_data.py
python run_experiment.py --mode dataset-check
python run_experiment.py --mode full
```

The full experiment uses validation-set reward selection and does not tune on the test set.

## Outputs

- `results/final_results.csv`
- `results/results_by_seed.csv`
- `results/statistical_tests.csv`
- `results/ablation_results.csv`
- `results/sensitivity_results.csv`
- `results/robustness_results.csv`
- `results/dataset_statistics.csv`
- `results/model_selection.csv`
- `figures/*.png`
- `paper/*.md`
- `paper/tables/*.csv`
- `FINAL_AUDIT.md`

## Research Integrity

The experiment does not target a predetermined savings percentage. It reports actual outputs, including negative or statistically non-estimable comparisons.
