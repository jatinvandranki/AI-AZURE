import pandas as pd
import pytest

from src.preprocessing import add_workload_features, clean_cpu_chunk, temporal_split


def test_clean_cpu_chunk_reports_invalid_and_duplicates():
    chunk = pd.DataFrame({
        0: [0, 0, 300, 600],
        1: ["vm1", "vm1", "vm2", None],
        4: [50, 50, 120, 10],
    })
    cleaned, stats = clean_cpu_chunk(chunk)
    assert len(cleaned) == 1
    assert cleaned["cpu_util"].iloc[0] == 0.5
    assert stats.duplicate_observations == 1
    assert stats.invalid_cpu_values == 1
    assert stats.missing_values == 1


def test_temporal_split_preserves_order():
    trace = pd.DataFrame({"step": range(10), "cpu_demand": range(10), "vm_count": [1] * 10, "mean_cpu_util": [0.1] * 10, "p95_cpu_util": [0.1] * 10, "max_cpu_util": [0.1] * 10})
    trace = add_workload_features(trace)
    train, validation, test = temporal_split(trace, 0.6, 0.2, 0.2)
    assert train["step"].max() < validation["step"].min() < test["step"].min()


def test_temporal_split_rejects_bad_ratios():
    with pytest.raises(ValueError):
        temporal_split(pd.DataFrame({"step": range(10)}), 0.5, 0.2, 0.2)

