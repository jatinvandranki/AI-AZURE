import numpy as np
import pandas as pd

import run_experiment
from src.agents import make_agents
from src.baselines import baseline_action
from src.config import load_config
from src.statistics import paired_wilcoxon


def test_agents_have_specialized_features():
    config = load_config(mode="quick")
    resource, energy = make_agents(config, 101)
    assert resource.features != energy.features
    assert resource.actions != energy.actions


def test_greedy_baseline_provisions_for_target_utilization():
    config = load_config(mode="quick")
    state = {"demand": 0.7}
    action = baseline_action("Greedy", state, 10, config)
    assert action.target_servers >= int(np.ceil(0.7 * config.get("max_servers") / 0.70))


def test_paired_stats_reports_not_estimable_for_zero_differences():
    rows = []
    for method in ["AI", "StaticOver"]:
        for i in range(3):
            rows.append({"method": method, "seed": 101, "interval": i, "energy_wh": 1.0})
    for method in ["Threshold", "Packing", "Greedy"]:
        for i in range(3):
            rows.append({"method": method, "seed": 101, "interval": i, "energy_wh": 2.0})
    out = paired_wilcoxon(pd.DataFrame(rows))
    assert "NOT ESTIMABLE" in set(out["test_statistic"])


def test_sensitivity_trains_each_seed_independently(monkeypatch):
    config = load_config(mode="quick")
    recorded = []

    def fake_train_policy(trace, cfg, seed, variant="full", reward_override=None):
        recorded.append(seed)
        return None, None, pd.DataFrame([{"episode": 1, "reward": 0.0, "epsilon": 0.1, "q_table_size": 0, "visited_states": 0, "unique_actions": 1, "policy_stability": 1.0}]), pd.DataFrame([{"episode": 1, "resource_states": 0, "energy_states": 0, "resource_q_std": 0.0, "energy_q_std": 0.0, "policy_stability": 1.0}])

    monkeypatch.setattr(run_experiment, "train_policy", fake_train_policy)
    monkeypatch.setattr(run_experiment, "evaluate_method", lambda *args, **kwargs: pd.DataFrame([{"method": "AI", "seed": args[3], "variant": args[5] if len(args) > 5 else "full", "energy_wh": 1.0, "workload": 1.0, "sla_violation": 0, "response_time_ms": 100.0, "utilization": 0.5, "active_servers": 10, "overload": 0.0, "churn": 0, "server_activation_count": 0, "server_deactivation_count": 0, "vm_migration_count": 0, "consolidation_count": 0, "training_seed": args[3], "evaluation_seed": args[3]}]))

    run_experiment.run_sensitivity(pd.DataFrame({"normalized_demand": [0.5, 0.5]}), pd.DataFrame({"normalized_demand": [0.5, 0.5]}), config)
    assert sorted(recorded) == sorted(config.get("seeds") * len(run_experiment.REWARD_PRESETS))


def test_paired_wilcoxon_aggregates_seed_runs_by_interval():
    rows = []
    for interval in [0, 1]:
        for seed in [101, 202]:
            rows.append({"method": "AI", "seed": seed, "interval": interval, "energy_wh": 10.0 + seed / 1000.0})
            rows.append({"method": "StaticOver", "seed": seed, "interval": interval, "energy_wh": 12.0 + seed / 1000.0})
    out = paired_wilcoxon(pd.DataFrame(rows))
    assert out.iloc[0]["n_pairs"] == 2

