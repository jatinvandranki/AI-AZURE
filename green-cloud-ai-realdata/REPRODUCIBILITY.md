# Reproducibility

## Environment

Use Python 3.11 or newer.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Commands

```bash
python run_experiment.py --mode quick
python run_experiment.py --mode standard
python run_experiment.py --mode dataset-check
python run_experiment.py --mode full
```

Only `--mode full` should be used for final paper results. If the local dataset is too small, full mode fails loudly rather than producing misleading final claims.

## Seeds

Default evaluation seeds are `101, 202, 303, 404, 505`. Seeds affect Q-learning exploration and training trajectory.

## Validation Selection

Reward configurations are evaluated on the validation split only. The selected configuration is saved to `results/model_selection.csv`, then the final model is trained on train plus validation and evaluated once on the untouched test split.

## Output Verification

The final pipeline writes CSV metrics under `results/`, 300 DPI figures under `figures/`, generated paper text under `paper/`, and an end-to-end `FINAL_AUDIT.md`. Energy values are model-based estimates from the documented linear power model, not direct Azure data-center measurements.
