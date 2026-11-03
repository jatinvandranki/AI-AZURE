from __future__ import annotations

import numpy as np
import pandas as pd
import math
try:
    from scipy import stats
except ModuleNotFoundError:
    stats = None


def bootstrap_ci(values, rng, samples: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    means = [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(samples)]
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted([(p, i) for i, p in enumerate(p_values) if np.isfinite(p)])
    adjusted = [np.nan] * len(p_values)
    running = 0.0
    m = len(indexed)
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, p * (m - rank)))
        adjusted[i] = running
    return adjusted


def paired_wilcoxon(timeseries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ai = timeseries[timeseries["method"] == "AI"].copy()
    for baseline in ["StaticOver", "Threshold", "Packing", "Greedy"]:
        other = timeseries[timeseries["method"] == baseline].copy()
        ai_interval = ai.groupby("interval", as_index=False)["energy_wh"].mean().rename(columns={"energy_wh": "ai_mean_energy"})
        other_interval = other.groupby("interval", as_index=False)["energy_wh"].mean().rename(columns={"energy_wh": "baseline_mean_energy"})
        paired = ai_interval.merge(other_interval, on="interval", how="inner")
        if paired.empty:
            rows.append({
                "comparison": f"AI vs {baseline}",
                "n_pairs": 0,
                "test_statistic": "NOT ESTIMABLE",
                "raw_p_value": np.nan,
                "effect_size": np.nan,
                "confidence_interval": "NOT ESTIMABLE",
                "significant_05": False,
            })
            continue
        diff = paired["ai_mean_energy"] - paired["baseline_mean_energy"].to_numpy()
        diff = np.asarray(diff, dtype=float)
        non_zero = diff[~np.isclose(diff, 0.0)]
        if len(non_zero) < 2:
            rows.append({
                "comparison": f"AI vs {baseline}",
                "n_pairs": int(len(diff)),
                "test_statistic": "NOT ESTIMABLE",
                "raw_p_value": np.nan,
                "effect_size": np.nan,
                "confidence_interval": "NOT ESTIMABLE",
                "significant_05": False,
            })
            continue
        stat, p = _wilcoxon(non_zero)
        low, high = bootstrap_ci(non_zero, np.random.default_rng(123), samples=1000)
        effect = float(np.mean(non_zero) / np.std(non_zero, ddof=1)) if np.std(non_zero, ddof=1) else np.nan
        rows.append({
            "comparison": f"AI vs {baseline}",
            "n_pairs": int(len(diff)),
            "test_statistic": float(stat),
            "raw_p_value": float(p),
            "effect_size": effect,
            "confidence_interval": f"[{low:.6f}, {high:.6f}]",
            "significant_05": False,
        })
    df = pd.DataFrame(rows)
    df["holm_adjusted_p_value"] = holm_adjust(df["raw_p_value"].tolist())
    df["significant_05"] = df["holm_adjusted_p_value"].fillna(1.0) < 0.05
    return df[["comparison", "n_pairs", "test_statistic", "raw_p_value", "holm_adjusted_p_value", "effect_size", "confidence_interval", "significant_05"]]


def metric_confidence_intervals(timeseries: pd.DataFrame, config) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    for method, group in timeseries.groupby("method"):
        for metric in ["energy_wh", "sla_violation", "response_time_ms", "utilization", "active_servers"]:
            low, high = bootstrap_ci(group[metric], rng, samples=int(config.get("bootstrap_samples")))
            rows.append({"method": method, "metric": metric, "mean": float(group[metric].mean()), "ci95_low": low, "ci95_high": high, "method_ci": "bootstrap_percentile"})
    return pd.DataFrame(rows)


def _wilcoxon(non_zero: np.ndarray) -> tuple[float, float]:
    if stats is not None:
        return stats.wilcoxon(non_zero, zero_method="wilcox")
    abs_diff = np.abs(non_zero)
    order = np.argsort(abs_diff)
    ranks = np.empty(len(abs_diff), dtype=float)
    ranks[order] = np.arange(1, len(abs_diff) + 1)
    w_plus = float(ranks[non_zero > 0].sum())
    w_minus = float(ranks[non_zero < 0].sum())
    statistic = min(w_plus, w_minus)
    n = len(non_zero)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    if var <= 0:
        return statistic, np.nan
    z = (statistic - mean) / np.sqrt(var)
    p = float(math.erfc(abs(z) / np.sqrt(2)))
    return statistic, p
