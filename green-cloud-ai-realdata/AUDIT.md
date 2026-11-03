# Audit

## Current Architecture

The original project was a flat Python prototype with `config.py`, `data_loader.py`, `environment.py`, `agents.py`, `experiment.py`, and `run_experiment.py`. Results and plots were produced directly by the runner.

## Current Dataset Pipeline

The original loader read Azure V2 VM metadata and CPU shards from `data/azure_v2`. CPU readings were interpreted as percentages and divided by 100, which is correct for the documented Azure CPU reading columns. The prior implementation concatenated per-shard chunks into memory and defaulted to whatever local shards existed.

## Current RL Algorithm

The project used tabular Q-learning with epsilon-greedy exploration, epsilon decay, learning rate, and discount factor constants.

## Current State Space

The previous state included normalized demand, previous demand, moving average, demand volatility, active capacity, SLA pressure, time fraction, and trend.

## Current Action Space

The Resource Allocation Agent selected scale-down, hold, or scale-up. The Energy Optimization Agent selected hold or consolidate.

## Current Reward

The reward penalized estimated energy, SLA violation, response latency, overload, and churn while rewarding utilization.

## Current Energy Model

The simulator used `P(u) = P_idle + (P_max - P_idle) * u` and integrated over five-minute intervals.

## Current SLA Model

SLA violation was based on response time exceeding a target threshold. The revised model also treats overload or unmet demand as violations.

## Current Latency Model

Latency increased with utilization and overload, with a small random term in the original environment. The revised environment is deterministic for fair paired comparisons.

## Current Baselines

The original runner included StaticOver, Threshold, Packing, and Greedy-like methods, but earlier output also showed runs without the Greedy baseline.

## Current Statistical Methodology

The previous pipeline mixed paired t-tests and Wilcoxon tests and reported strong claims from a very small held-out sample. The revised pipeline uses paired interval-level Wilcoxon tests with Holm correction and reports `NOT ESTIMABLE` when the test is not meaningful.

## Current Weaknesses

- Flat constants scattered across modules.
- No 60/20/20 train/validation/test split.
- Loader could use one shard and produce only 45 intervals.
- Final-paper claims were possible with insufficient local data.
- Results text contained risk of overclaiming.
- Test suite was absent.
- Dataset files were not robustly excluded by project `.gitignore`.
- Evaluation stochasticity could obscure paired comparisons.

## Recommended Improvements

- Added centralized YAML configuration and quick/standard/full modes.
- Added chunked dataset discovery and data quality reporting.
- Added temporal split validation and leakage checks.
- Added two specialized agents with separate feature sets and explicit coordination.
- Added fair shared environment for all methods.
- Added required baselines, ablations, reward sensitivity, robustness slices, confidence intervals, and paired tests.
- Added generated paper-ready Markdown and CSV tables from actual outputs.
- Added required `figures/` output names and `FINAL_AUDIT.md` generation from actual run data.
