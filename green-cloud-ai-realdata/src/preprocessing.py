from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class PreprocessingStats:
    raw_rows: int = 0
    valid_rows: int = 0
    discarded_rows: int = 0
    missing_values: int = 0
    duplicate_observations: int = 0
    invalid_cpu_values: int = 0
    malformed_records: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def clean_cpu_chunk(chunk: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingStats]:
    stats = PreprocessingStats(raw_rows=len(chunk))
    if chunk.shape[1] < 3:
        stats.malformed_records = len(chunk)
        stats.discarded_rows = len(chunk)
        return pd.DataFrame(columns=["timestamp", "vm_id", "cpu_util"]), stats

    chunk = chunk.iloc[:, :3].copy()
    chunk.columns = ["timestamp", "vm_id", "cpu_util_pct"]
    stats.missing_values = int(chunk.isna().sum().sum())
    chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
    chunk["cpu_util_pct"] = pd.to_numeric(chunk["cpu_util_pct"], errors="coerce")
    before = len(chunk)
    chunk = chunk.dropna(subset=["timestamp", "vm_id", "cpu_util_pct"])
    stats.malformed_records += before - len(chunk)

    valid_cpu = chunk["cpu_util_pct"].between(0, 100) & np.isfinite(chunk["cpu_util_pct"])
    stats.invalid_cpu_values = int((~valid_cpu).sum())
    chunk = chunk[valid_cpu]
    chunk = chunk[(chunk["timestamp"] >= 0) & np.isfinite(chunk["timestamp"])]
    stats.duplicate_observations = int(chunk.duplicated(["timestamp", "vm_id"]).sum())
    chunk = chunk.drop_duplicates(["timestamp", "vm_id"])
    chunk["cpu_util"] = chunk["cpu_util_pct"] / 100.0
    chunk = chunk[["timestamp", "vm_id", "cpu_util"]]
    stats.valid_rows = len(chunk)
    stats.discarded_rows = stats.raw_rows - stats.valid_rows
    return chunk, stats


def add_workload_features(intervals: pd.DataFrame) -> pd.DataFrame:
    out = intervals.sort_values("step").reset_index(drop=True).copy()
    demand = out["cpu_demand"].to_numpy(dtype=float)
    scale = max(float(np.nanpercentile(demand, 99)), 1e-9)
    out["normalized_demand"] = np.clip(demand / scale, 0.0, 1.0)
    out["previous_demand"] = out["normalized_demand"].shift(1).fillna(out["normalized_demand"].iloc[0])
    out["moving_average_demand"] = out["normalized_demand"].rolling(5, min_periods=1).mean()
    out["demand_volatility"] = out["normalized_demand"].rolling(5, min_periods=2).std().fillna(0.0)
    slope = out["normalized_demand"] - out["previous_demand"]
    out["trend"] = np.select(
        [slope > 0.05, slope < -0.05, out["demand_volatility"] > 0.12],
        ["rising", "falling", "volatile"],
        default="stable",
    )
    out["timestamp"] = out["step"] * out.attrs.get("interval_seconds", 300)
    return out


def temporal_split(trace: pd.DataFrame, train_ratio: float, validation_ratio: float, test_ratio: float):
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("Train/validation/test ratios must sum to 1.0")
    n = len(trace)
    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)
    if train_end <= 0 or validation_end <= train_end or validation_end >= n:
        raise ValueError(f"Invalid temporal split for {n} intervals")
    train = trace.iloc[:train_end].reset_index(drop=True)
    validation = trace.iloc[train_end:validation_end].reset_index(drop=True)
    test = trace.iloc[validation_end:].reset_index(drop=True)
    if train["step"].max() >= validation["step"].min() or validation["step"].max() >= test["step"].min():
        raise ValueError("Temporal split leakage detected")
    return train, validation, test

