from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def validate_inputs(trace: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, report: dict, config) -> None:
    failures = []
    if report["files_used"] <= 0:
        failures.append("No trace files used")
    if len(trace) < int(config.mode_settings.get("min_intervals")):
        msg = f"Only {len(trace)} intervals available; {config.mode} requires at least {config.mode_settings.get('min_intervals')}"
        if config.mode_settings.get("strict_publication"):
            failures.append(msg)
        else:
            print(f"WARNING: {msg}")
    if train["step"].max() >= validation["step"].min() or validation["step"].max() >= test["step"].min():
        failures.append("Temporal split leakage detected")
    if test.duplicated(["step"]).any():
        failures.append("Duplicate test intervals detected")
    if trace.isna().any().any():
        failures.append("NaN values in workload trace")
    if (trace["normalized_demand"] < 0).any() or (trace["normalized_demand"] > 1).any():
        failures.append("Impossible normalized demand values")
    if failures:
        raise ValueError("Validation failed: " + "; ".join(failures))


def validate_outputs(timeseries: pd.DataFrame, summaries: pd.DataFrame, expected_files: list[Path], config) -> None:
    failures = []
    if timeseries[["energy_wh", "utilization", "active_servers"]].isna().any().any():
        failures.append("NaN metrics in interval results")
    if (timeseries["energy_wh"] < 0).any():
        failures.append("Negative energy values")
    if (timeseries["utilization"] < 0).any() or (timeseries["utilization"] > 1).any():
        failures.append("Impossible utilization values")
    if (timeseries["active_servers"] < 1).any() or (timeseries["active_servers"] > int(config.get("max_servers"))).any():
        failures.append("Impossible active server count")
    if not np.allclose(timeseries["sla_compliance"], 1 - timeseries["sla_violation"]):
        failures.append("SLA compliance is inconsistent with violation rate")
    methods = set(timeseries["method"].unique())
    for seed, group in timeseries.groupby("seed"):
        counts = group.groupby("method")["interval"].nunique()
        if len(set(counts)) != 1:
            failures.append(f"Methods did not see identical interval counts for seed {seed}")
    missing = [str(p) for p in expected_files if not p.exists()]
    if missing:
        failures.append("Missing expected outputs: " + ", ".join(missing))
    if failures:
        raise ValueError("Output validation failed: " + "; ".join(failures))
