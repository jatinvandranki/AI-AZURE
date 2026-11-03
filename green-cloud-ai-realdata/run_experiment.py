#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config
from src.data_loader import build_workload, discover_trace_files
from src.dataset_requirements import inventory_data_dir, required_file_names
from src.evaluation import evaluate_all, evaluate_method, summarize
from src.preprocessing import temporal_split
from src.statistics import metric_confidence_intervals, paired_wilcoxon
from src.training import REWARD_PRESETS, train_policy
from src.validation import validate_inputs, validate_outputs
from src.visualization import save_figures


def parse_args():
    parser = argparse.ArgumentParser(description="Trace-driven green cloud multi-agent AI experiment")
    parser.add_argument("--mode", choices=["dataset-check", "quick", "standard", "full"], default="quick")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--allow-small-full", action="store_true")
    return parser.parse_args()


def dataset_check(config):
    ensure_dirs(config)
    try:
        trace, report_obj = build_workload(config)
        current_intervals = report_obj.workload_intervals
        report = report_obj.to_dict()
        trace.to_csv(config.results_dir / "processed_workload.csv", index=False)
        pd.DataFrame([report]).to_csv(config.results_dir / "dataset_statistics.csv", index=False)
        write_dataset_manifest(config)
    except Exception as exc:
        print(f"Dataset processing failed during check: {exc}")
        current_intervals = 0
    requirement = inventory_data_dir(config, current_intervals=current_intervals)
    missing = required_file_names(config, requirement)
    out = {**requirement.to_dict(), "missing_files": missing}
    (config.results_dir / "dataset_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nDATASET CHECK\n-------------")
    print(f"Data directory: {requirement.data_dir}")
    print(f"VM table present: {requirement.vm_table_present} ({requirement.vm_table_name})")
    print(f"CPU files present: {requirement.cpu_files_present}")
    print(f"Current genuine chronological intervals: {requirement.current_intervals}")
    print(f"Publication target intervals: {requirement.target_intervals}")
    print(f"Estimated CPU files required total: {requirement.required_cpu_files_total}")
    print(f"Additional CPU files needed: {requirement.additional_cpu_files_needed}")
    print(f"Approximate additional download: {requirement.approximate_required_download_gb} GB")
    if missing:
        print("\nDownload these files:")
        for name in missing:
            print(f"- {name}")
    print("\nStatus:", "SUFFICIENT" if requirement.sufficient else "INSUFFICIENT")
    return 0 if requirement.sufficient else 2


def dataset_manifest(config) -> list[dict]:
    files = discover_trace_files(config.data_dir)
    table = sorted(config.data_dir.glob("*vmtable*.csv*"))
    all_files = [*table[:1], *files]
    return [{"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime} for p in all_files]


def write_dataset_manifest(config) -> None:
    (config.results_dir / "dataset_manifest.json").write_text(
        json.dumps({"interval_seconds": config.get("interval_seconds"), "files": dataset_manifest(config)}, indent=2),
        encoding="utf-8",
    )


def load_cached_workload(config):
    workload_path = config.results_dir / "processed_workload.csv"
    stats_path = config.results_dir / "dataset_statistics.csv"
    manifest_path = config.results_dir / "dataset_manifest.json"
    if not (workload_path.exists() and stats_path.exists() and manifest_path.exists()):
        return None
    cached = json.loads(manifest_path.read_text(encoding="utf-8"))
    if cached.get("interval_seconds") != config.get("interval_seconds"):
        return None
    if cached.get("files") != dataset_manifest(config):
        return None
    trace = pd.read_csv(workload_path)
    report = pd.read_csv(stats_path).iloc[0].to_dict()
    return trace, report


def ensure_dirs(config):
    for path in [config.results_dir, config.plots_dir, config.paper_dir, config.paper_dir / "tables"]:
        path.mkdir(exist_ok=True)


def train_and_evaluate(train, test, config, variant="full", reward_override=None):
    rows = []
    summaries = []
    last_resource = None
    last_energy = None
    last_history = None
    last_diagnostics = None
    for seed in config.get("seeds"):
        resource, energy, history, diagnostics = train_policy(train, config, seed=seed, variant=variant, reward_override=reward_override)
        last_resource, last_energy, last_history, last_diagnostics = resource, energy, history, diagnostics
        timeseries, summary = evaluate_all(test, config, resource, energy, variant=variant, evaluation_seed=seed, training_seed=seed)
        rows.append(timeseries)
        summaries.append(summary)
    return last_resource, last_energy, last_history, last_diagnostics, pd.concat(rows, ignore_index=True), pd.concat(summaries, ignore_index=True)


def run_sensitivity(train, validation, config):
    rows = []
    for name, override in REWARD_PRESETS.items():
        for seed in config.get("seeds"):
            resource, energy, _, _ = train_policy(train, config, seed=seed, reward_override=override)
            out = evaluate_method(validation, config, "AI", seed, resource, energy, training_seed=seed)
            row = summarize(out)
            row["configuration"] = name
            rows.append(row)
    return pd.DataFrame(rows)


def select_reward_configuration(sensitivity: pd.DataFrame, config):
    grouped = sensitivity.groupby("configuration").mean(numeric_only=True).copy()
    grouped["energy_norm"] = grouped["energy_per_workload"] / max(grouped["energy_per_workload"].max(), 1e-9)
    grouped["latency_norm"] = grouped["p95_response_ms"] / max(float(config.get("sla_threshold_ms")), 1e-9)
    grouped["selection_score"] = (
        grouped["energy_norm"]
        + 5.0 * grouped["sla_violation_rate"]
        + 0.5 * grouped["latency_norm"]
        + 0.05 * grouped["total_churn"] / max(float(grouped["total_churn"].max()), 1.0)
    )
    selected = str(grouped["selection_score"].idxmin())
    table = grouped.reset_index()
    return selected, REWARD_PRESETS[selected], table


def run_ablation(train, test, config, reward_override=None):
    variants = {
        "Full Multi-Agent AI": "full",
        "AI without workload trend": "no_trend",
        "AI without energy reward": "no_energy",
        "Resource Allocation Agent only": "resource_only",
        "Energy Optimization Agent only": "energy_only",
        "AI without coordination": "no_coordination",
    }
    rows = []
    for label, variant in variants.items():
        for seed in config.get("seeds"):
            resource, energy, _, _ = train_policy(train, config, seed=seed, variant=variant, reward_override=reward_override)
            out = evaluate_method(test, config, "AI", seed, resource, energy, variant, training_seed=seed)
            row = summarize(out)
            row["ablation"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def run_robustness(test, config, resource, energy):
    scenarios = {
        "high_demand": test[test["normalized_demand"] >= test["normalized_demand"].quantile(0.75)],
        "low_demand": test[test["normalized_demand"] <= test["normalized_demand"].quantile(0.25)],
        "volatile": test[test["demand_volatility"] >= test["demand_volatility"].quantile(0.75)],
        "spikes": test[test["trend"] == "rising"],
    }
    rows = []
    for scenario, subset in scenarios.items():
        if subset.empty:
            continue
        subset = subset.reset_index(drop=True)
        for seed in config.get("seeds"):
            out = evaluate_method(subset, config, "AI", seed, resource, energy)
            rows.append({"scenario": scenario, **summarize(out)})
    return pd.DataFrame(rows)


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in summary.columns if c not in {"method", "seed", "variant"}]
    return summary.groupby("method")[metrics].agg(["mean", "std"]).round(6)


def write_training_plots(history, plots_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["episode"], history["reward"], linewidth=1.5)
    ax.set_title("Training reward curve"); ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    fig.tight_layout(); fig.savefig(plots_dir / "training_reward.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["episode"], history["epsilon"], linewidth=1.5)
    ax.set_title("Epsilon decay"); ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon")
    fig.tight_layout(); fig.savefig(plots_dir / "epsilon_curve.png", dpi=300); plt.close(fig)


def write_paper(config, report, summary, stats_df, ablation, sensitivity, robustness):
    paper = config.paper_dir
    means = summary.groupby("method").mean(numeric_only=True)
    ai = means.loc["AI"]
    greedy = means.loc["Greedy"] if "Greedy" in means.index else means.drop("AI").iloc[0]
    energy_delta = (greedy["energy_wh_total"] - ai["energy_wh_total"]) / max(greedy["energy_wh_total"], 1e-9) * 100
    label = "PUBLICATION CANDIDATE" if config.mode == "full" else f"NON-PUBLICATION {config.mode.upper()} DEBUG RUN"
    (paper / "abstract_results.md").write_text(
        f"Run label: {label}. The proposed multi-agent AI was evaluated on held-out chronological Azure CPU trace intervals. "
        f"Against Greedy, the estimated-energy difference was {energy_delta:.2f}% with AI SLA violation rate {ai['sla_violation_rate']:.4f}. "
        f"Energy is model-based, not measured facility power.\n",
        encoding="utf-8",
    )
    (paper / "methodology.md").write_text(
        "## Methodology\n\nThe experiment uses Microsoft Azure Public Dataset V2 CPU readings. CPU values are interpreted as percentages and divided by 100. "
        "Workload intervals are constructed chronologically with no random shuffling before the 60/20/20 train/validation/test split. "
        "All methods use the same homogeneous-server simulator, linear power model, latency model, and SLA definition.\n\n"
        "Energy is computed as P(u) = P_idle + (P_max - P_idle) * u, integrated over each interval. "
        "This is a model-based estimate of energy consumption and not direct measurement of Azure data-center energy.\n",
        encoding="utf-8",
    )
    (paper / "results.md").write_text(markdown_table(means.reset_index()) + "\n", encoding="utf-8")
    (paper / "discussion.md").write_text(
        f"## Discussion\n\nRun label: {label}. The proposed system is evaluated against static overprovisioning, threshold scaling, packing, and a greedy energy-aware heuristic using paired interval-level observations. "
        "Negative or nonsignificant results should be reported as such.\n",
        encoding="utf-8",
    )
    (paper / "limitations.md").write_text(
        "## Limitations\n\nThe Azure trace exposes CPU utilization, not direct data-center electricity or end-user latency. Energy is estimated from a transparent server power model. "
        "The simulator uses homogeneous servers, modeled queueing/latency, and simplified migration costs. External validity depends on whether selected shards represent the target deployment. "
        "The Q-learning agents use discretized state features and may not generalize to production control planes without further validation. "
        "Training cost grows with trace length, candidate configurations, and ablation runs.\n",
        encoding="utf-8",
    )
    (paper / "reproducibility.md").write_text(
        "## Reproducibility\n\nRun `python run_experiment.py --mode dataset-check` followed by `python run_experiment.py --mode full`. "
        f"The configured seeds are {config.get('seeds')}. Raw Azure CSV files are read from a relative data directory and are excluded from git. "
        "Reward configuration is selected on the validation split only; the test split is evaluated after selection.\n",
        encoding="utf-8",
    )
    tables = {
        "dataset_statistics": pd.DataFrame([report]),
        "final_results": aggregate(summary).reset_index(),
        "statistical_tests": stats_df,
        "ablation_results": ablation,
        "sensitivity_results": sensitivity,
        "robustness_results": robustness,
    }
    for name, df in tables.items():
        df.to_csv(paper / "tables" / f"{name}.csv", index=False)


def publication_safe_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    out = stats_df.copy()
    text = "Not estimable under the selected statistical procedure."
    for col in ["test_statistic", "raw_p_value", "holm_adjusted_p_value", "effect_size", "confidence_interval"]:
        out[col] = out[col].apply(lambda value: text if pd.isna(value) else value)
    return out


def write_final_audit(config, report, train, validation, test, selected_name, summary, stats_df, ablation, sensitivity, robustness):
    means = summary.groupby("method").mean(numeric_only=True).reset_index()
    audit = [
        "# Final Audit",
        "",
        "## Dataset Files Used",
        report.get("files_used_names", ""),
        "",
        "## Dataset Counts",
        f"Raw rows: {report.get('raw_rows')}",
        f"Valid rows: {report.get('valid_rows')}",
        f"VMs: {report.get('unique_vm_identifiers')}",
        f"Chronological intervals: {report.get('workload_intervals')}",
        f"Train/validation/test sizes: {len(train)}/{len(validation)}/{len(test)}",
        "",
        "## Seeds And Training",
        f"Seeds: {config.get('seeds')}",
        f"Episodes: {config.mode_settings.get('num_episodes')}",
        f"Selected model/configuration: {selected_name}",
        "",
        "## Main Results",
        means.to_markdown(index=False),
        "",
        "## Statistical Results",
        stats_df.to_markdown(index=False),
        "",
        "## Ablation Results",
        ablation.groupby("ablation").mean(numeric_only=True).reset_index().to_markdown(index=False),
        "",
        "## Sensitivity Results",
        sensitivity.groupby("configuration").mean(numeric_only=True).reset_index().to_markdown(index=False),
        "",
        "## Robustness Results",
        "No non-empty robustness slices available." if robustness.empty else robustness.groupby("scenario").mean(numeric_only=True).reset_index().to_markdown(index=False),
        "",
        "## Limitations",
        "Energy is a model-based estimate, not measured Azure data-center power. The simulator, homogeneous server assumptions, modeled latency, Azure trace coverage, Q-learning discretization, and deployment gap limit generalization.",
        "",
        "## Reproduction Commands",
        "```bash",
        "python run_experiment.py --mode dataset-check",
        "python run_experiment.py --mode full",
        "pytest",
        "python -m compileall src tests",
        "```",
        "",
        "## Remaining Warnings",
        "Do not interpret estimated energy as facility measurements. Do not claim state of the art without an external benchmark comparison.",
    ]
    (config.root / "FINAL_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")


def final_report(report, train, validation, test, history, summary, stats_df, ablation, robustness):
    ai = summary[summary["method"] == "AI"].mean(numeric_only=True)
    print("\nDATASET\n-------")
    print(f"Files used: {report['files_used']} of {report['files_discovered']}")
    print(f"Raw rows: {report['raw_rows']}")
    print(f"Valid rows: {report['valid_rows']}")
    print(f"Unique VMs: {report['unique_vm_identifiers']}")
    print(f"Time range: {report['time_start']} to {report['time_end']}")
    print(f"Total intervals: {report['workload_intervals']}")
    print("\nSPLIT\n-----")
    print(f"Train: {len(train)}  Validation: {len(validation)}  Test: {len(test)}")
    print("\nTRAINING\n--------")
    print(f"Episodes: {len(history)}")
    print(f"Seeds: {summary['seed'].drop_duplicates().tolist()}")
    print(f"Visited states: {int(history['visited_states'].iloc[-1])}")
    print(f"Q-table size: {int(history['q_table_size'].iloc[-1])}")
    print(f"Final epsilon: {history['epsilon'].iloc[-1]:.4f}")
    print("\nAI RESULTS\n----------")
    for label, value in [
        ("Energy", ai["energy_wh_total"]),
        ("Energy/workload", ai["energy_per_workload"]),
        ("SLA violation rate", ai["sla_violation_rate"]),
        ("SLA compliance", ai["sla_compliance_rate"]),
        ("Mean latency", ai["mean_response_ms"]),
        ("P95 latency", ai["p95_response_ms"]),
        ("Mean utilization", ai["mean_utilization"]),
        ("Mean active servers", ai["mean_active_servers"]),
        ("Mean overload", ai["mean_overload"]),
        ("Churn", ai["total_churn"]),
    ]:
        print(f"{label}: {value}")
    print("\nBASELINES\n---------")
    print(summary.groupby("method")["energy_wh_total"].mean().to_string())
    print("\nSTATISTICS\n----------")
    print(stats_df.to_string(index=False))
    print("\nABLATION\n--------")
    print(ablation.groupby("ablation")["energy_wh_total"].mean().to_string())
    print("\nROBUSTNESS\n----------")
    print("No non-empty robustness slices available." if robustness.empty else robustness.groupby("scenario")["energy_wh_total"].mean().to_string())


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    start = time.time()
    args = parse_args()
    load_mode = "full" if args.mode == "dataset-check" else args.mode
    config = load_config(mode=load_mode, data_dir=args.data_dir)
    if args.max_files is not None:
        config.values["modes"][args.mode]["max_files"] = args.max_files
    if args.max_rows is not None:
        config.values["modes"][args.mode]["max_rows"] = args.max_rows
    if args.allow_small_full and args.mode != "dataset-check":
        config.values["modes"][args.mode]["strict_publication"] = False
    if args.mode == "dataset-check":
        return dataset_check(config)
    ensure_dirs(config)

    cached = load_cached_workload(config)
    if cached is None:
        trace, report_obj = build_workload(config)
        report = report_obj.to_dict()
        trace.to_csv(config.results_dir / "processed_workload.csv", index=False)
        pd.DataFrame([report]).to_csv(config.results_dir / "dataset_statistics.csv", index=False)
        write_dataset_manifest(config)
    else:
        trace, report = cached
        print("Using validated cached workload from results/processed_workload.csv")
    train, validation, test = temporal_split(trace, config.get("train_ratio"), config.get("validation_ratio"), config.get("test_ratio"))
    validate_inputs(trace, train, validation, test, report, config)
    train.to_csv(config.results_dir / "train_workload.csv", index=False)
    validation.to_csv(config.results_dir / "validation_workload.csv", index=False)
    test.to_csv(config.results_dir / "test_workload.csv", index=False)

    sensitivity = run_sensitivity(train, validation, config)
    selected_name, selected_reward, model_selection = select_reward_configuration(sensitivity, config)
    model_selection.to_csv(config.results_dir / "model_selection.csv", index=False)
    final_train = pd.concat([train, validation], ignore_index=True)
    resource, energy, history, diagnostics, timeseries, summary = train_and_evaluate(final_train, test, config, reward_override=selected_reward)
    history.to_csv(config.results_dir / "training_history.csv", index=False)
    diagnostics.to_csv(config.results_dir / "training_diagnostics.csv", index=False)
    timeseries.to_csv(config.results_dir / "timeseries_results.csv", index=False)
    summary.to_csv(config.results_dir / "results_by_seed.csv", index=False)
    aggregate(summary).to_csv(config.results_dir / "final_results.csv")
    stats_df = publication_safe_stats(paired_wilcoxon(timeseries))
    stats_df.to_csv(config.results_dir / "statistical_tests.csv", index=False)
    metric_confidence_intervals(timeseries, config).to_csv(config.results_dir / "confidence_intervals.csv", index=False)
    sensitivity.to_csv(config.results_dir / "sensitivity_results.csv", index=False)
    ablation = run_ablation(final_train, test, config, reward_override=selected_reward)
    ablation.to_csv(config.results_dir / "ablation_results.csv", index=False)
    robustness = run_robustness(test, config, resource, energy)
    robustness.to_csv(config.results_dir / "robustness_results.csv", index=False)

    write_training_plots(history, config.plots_dir)
    save_figures(trace, timeseries, summary, ablation, sensitivity, robustness, config.plots_dir)
    expected = [config.results_dir / name for name in [
        "training_history.csv", "training_diagnostics.csv", "timeseries_results.csv", "results_by_seed.csv",
        "final_results.csv", "statistical_tests.csv", "ablation_results.csv", "sensitivity_results.csv",
        "robustness_results.csv", "dataset_statistics.csv", "model_selection.csv",
    ]]
    validate_outputs(timeseries, summary, expected, config)
    write_paper(config, report, summary, stats_df, ablation, sensitivity, robustness)
    write_final_audit(config, report, train, validation, test, selected_name, summary, stats_df, ablation, sensitivity, robustness)
    final_report(report, train, validation, test, history, summary, stats_df, ablation, robustness)
    print(f"\nVALIDATION SELECTION\n--------------------\nSelected reward configuration: {selected_name}")
    print(f"\nComplete in {(time.time() - start) / 60:.2f} minutes. Mode: {args.mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
