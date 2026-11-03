from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_figures(trace: pd.DataFrame, timeseries: pd.DataFrame, summary: pd.DataFrame, ablation: pd.DataFrame, sensitivity: pd.DataFrame, robustness: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    _architecture(out_dir / "architecture.png")
    _line(trace, "normalized_demand", "Dataset workload distribution", "Normalized demand", out_dir / "data_distribution.png")
    means = summary.groupby("method").mean(numeric_only=True)
    _bar(means, "energy_wh_total", "Estimated energy comparison", "Wh", out_dir / "energy_comparison.png")
    static = means.loc["StaticOver", "energy_wh_total"] if "StaticOver" in means.index else means["energy_wh_total"].max()
    reduction = (static - means["energy_wh_total"]) / static * 100
    _bar(reduction.to_frame("energy_reduction_pct"), "energy_reduction_pct", "Energy reduction vs StaticOver", "%", out_dir / "energy_reduction.png")
    _bar(means, "sla_violation_rate", "SLA comparison", "Violation rate", out_dir / "sla_comparison.png")
    _bar(means, "p95_response_ms", "Latency comparison", "P95 ms", out_dir / "latency_comparison.png")
    _bar(means, "mean_utilization", "Utilization comparison", "Mean utilization", out_dir / "utilization_comparison.png")
    for metric, title, filename in [
        ("active_servers", "Active servers over time", "active_servers.png"),
        ("energy_wh", "Energy consumption over time", "energy_timeseries.png"),
    ]:
        pivot = timeseries.groupby(["interval", "method"])[metric].mean().unstack()
        ax = pivot.plot(figsize=(9, 4), linewidth=1.5)
        ax.set_title(title); ax.set_xlabel("Test interval"); ax.set_ylabel(metric)
        ax.figure.tight_layout(); ax.figure.savefig(out_dir / filename, dpi=300); plt.close(ax.figure)
    _bar(ablation.groupby("ablation").mean(numeric_only=True), "energy_wh_total", "Ablation comparison", "Wh", out_dir / "ablation.png")
    _scatter(sensitivity, "energy_wh_total", "sla_violation_rate", "configuration", "Reward sensitivity", out_dir / "sensitivity.png")
    _scatter(summary, "energy_wh_total", "sla_violation_rate", "method", "Pareto frontier", out_dir / "pareto_frontier.png")
    _bar(robustness.groupby("scenario").mean(numeric_only=True), "energy_wh_total", "Robustness/generalization", "Wh", out_dir / "robustness.png")


def _architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    text = "Azure CPU traces -> quality checks -> workload features -> temporal split\nResource Allocation Agent <-> Energy Optimization Agent -> shared simulator\nFair baselines -> interval metrics -> paired statistics -> paper tables"
    ax.text(0.03, 0.72, text, fontsize=14, va="top")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def _line(df, metric, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df[metric].to_numpy(), linewidth=1.5)
    ax.set_title(title); ax.set_xlabel("Chronological interval"); ax.set_ylabel(ylabel)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def _bar(df, metric, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    df[metric].plot.bar(ax=ax, color="#3b7f62")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def _scatter(df, x, y, label, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    grouped = df.groupby(label).mean(numeric_only=True)
    ax.scatter(grouped[x], grouped[y], s=80, color="#3b7f62")
    for name, row in grouped.iterrows():
        ax.annotate(str(name), (row[x], row[y]))
    ax.set_title(title); ax.set_xlabel(x); ax.set_ylabel(y)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
