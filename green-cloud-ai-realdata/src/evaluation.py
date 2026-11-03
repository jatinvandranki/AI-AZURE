from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import baseline_action
from .environment import CloudAction, CloudEnvironment
from .training import coordinated_action


METHODS = ["StaticOver", "Threshold", "Packing", "Greedy", "AI"]


def evaluate_method(trace, config, method: str, seed: int, resource=None, energy=None, variant: str = "full", training_seed: int | None = None) -> pd.DataFrame:
    env = CloudEnvironment(trace, config, seed=seed, use_trend=variant != "no_trend")
    state = env.reset()
    capacity = int(config.get("initial_active_servers"))
    rows = []
    training_seed_value = training_seed if training_seed is not None else seed
    for _ in range(len(trace)):
        if method == "AI":
            if resource is None and energy is None:
                raise ValueError("AI evaluation requires at least one trained agent")
            action, _ = coordinated_action(resource, energy, state, capacity, config, variant, training=False)
            capacity = action.target_servers
            action = CloudAction(capacity, action.consolidate)
        else:
            action = baseline_action(method, state, capacity, config)
            capacity = action.target_servers
        state, done, info = env.step(action)
        info.update({
            "method": method,
            "seed": seed,
            "variant": variant,
            "training_seed": training_seed_value if method == "AI" else np.nan,
            "evaluation_seed": seed,
        })
        rows.append(info)
        if done:
            break
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    training_seed = df["training_seed"].iloc[0] if "training_seed" in df.columns else np.nan
    evaluation_seed = df["evaluation_seed"].iloc[0] if "evaluation_seed" in df.columns else int(df["seed"].iloc[0])
    return {
        "method": df["method"].iloc[0],
        "seed": int(df["seed"].iloc[0]),
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "variant": df["variant"].iloc[0],
        "energy_wh_total": float(df["energy_wh"].sum()),
        "energy_wh_mean": float(df["energy_wh"].mean()),
        "energy_per_workload": float(df["energy_wh"].sum() / max(df["workload"].sum(), 1e-9)),
        "sla_violations": int(df["sla_violation"].sum()),
        "sla_violation_rate": float(df["sla_violation"].mean()),
        "sla_compliance_rate": float(1.0 - df["sla_violation"].mean()),
        "mean_response_ms": float(df["response_time_ms"].mean()),
        "median_response_ms": float(df["response_time_ms"].median()),
        "p95_response_ms": float(df["response_time_ms"].quantile(0.95)),
        "p99_response_ms": float(df["response_time_ms"].quantile(0.99)),
        "mean_utilization": float(df["utilization"].mean()),
        "p95_utilization": float(df["utilization"].quantile(0.95)),
        "mean_active_servers": float(df["active_servers"].mean()),
        "peak_active_servers": int(df["active_servers"].max()),
        "server_activation_count": int(df["server_activation_count"].sum()),
        "server_deactivation_count": int(df["server_deactivation_count"].sum()),
        "vm_migration_count": int(df["vm_migration_count"].sum()),
        "consolidation_count": int(df["consolidation_count"].sum()),
        "total_churn": int(df["churn"].sum()),
        "mean_overload": float(df["overload"].mean()),
        "max_overload": float(df["overload"].max()),
    }


def evaluate_all(test, config, resource, energy, variant: str = "full", evaluation_seed: int | None = None, training_seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summaries = []
    seed = int(evaluation_seed) if evaluation_seed is not None else int(config.get("seeds", [0])[0])
    for method in METHODS:
        out = evaluate_method(test, config, method, seed, resource, energy, variant, training_seed=training_seed if training_seed is not None else seed)
        rows.append(out)
        summaries.append(summarize(out))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)
