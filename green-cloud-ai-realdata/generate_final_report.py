from __future__ import annotations

from pathlib import Path

import pandas as pd

from run_experiment import (
    aggregate,
    write_final_audit,
    write_paper,
)
from src.config import load_config


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    config = load_config(mode="full")

    report = read_csv(RESULTS / "dataset_statistics.csv")
    report = report.iloc[0].to_dict() if not report.empty else {}

    summary = read_csv(RESULTS / "results_by_seed.csv")
    stats_df = read_csv(RESULTS / "statistical_tests.csv")
    ablation = read_csv(RESULTS / "ablation_results.csv")
    sensitivity = read_csv(RESULTS / "sensitivity_results.csv")
    robustness = read_csv(RESULTS / "robustness_results.csv")
    final_results = read_csv(RESULTS / "final_results.csv")
    model_selection = read_csv(RESULTS / "model_selection.csv")

    train = read_csv(RESULTS / "train_workload.csv")
    validation = read_csv(RESULTS / "validation_workload.csv")
    test = read_csv(RESULTS / "test_workload.csv")

    selected_name = ""
    if not model_selection.empty and "selection_score" in model_selection.columns:
        selected_name = str(model_selection.loc[model_selection["selection_score"].idxmin(), "configuration"])

    # Rebuild the final audit and paper artifacts from the existing run outputs only.
    write_paper(config, report, summary, stats_df, ablation, sensitivity, robustness)
    write_final_audit(
        config,
        report,
        train,
        validation,
        test,
        selected_name,
        summary,
        stats_df,
        ablation,
        sensitivity,
        robustness,
    )

    tables_dir = PAPER / "tables"
    tables_dir.mkdir(exist_ok=True)

    for csv_name, df in {
        "dataset_statistics.csv": pd.DataFrame([report]),
        "final_results.csv": final_results,
        "statistical_tests.csv": stats_df,
        "ablation_results.csv": ablation,
        "sensitivity_results.csv": sensitivity,
        "robustness_results.csv": robustness,
    }.items():
        df.to_csv(tables_dir / csv_name, index=False)

    print("Recovered final reporting artifacts from existing result files.")
    print(f"Selected reward config: {selected_name}")
    print(f"Paper directory: {PAPER}")


if __name__ == "__main__":
    main()
