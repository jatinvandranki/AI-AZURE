import pandas as pd

from src.config import load_config
from src.environment import CloudAction, CloudEnvironment, compute_reward
from src.preprocessing import add_workload_features


def sample_trace():
    df = pd.DataFrame({
        "step": range(6),
        "cpu_demand": [10, 20, 30, 40, 50, 60],
        "vm_count": [5] * 6,
        "mean_cpu_util": [0.4] * 6,
        "p95_cpu_util": [0.7] * 6,
        "max_cpu_util": [0.9] * 6,
    })
    return add_workload_features(df)


def test_energy_model_is_linear_and_positive():
    config = load_config(mode="quick")
    env = CloudEnvironment(sample_trace(), config)
    low = env.estimate_energy(10, 0.2)
    high = env.estimate_energy(10, 0.8)
    assert high > low > 0


def test_sla_violation_matches_compliance():
    config = load_config(mode="quick")
    env = CloudEnvironment(sample_trace(), config)
    env.reset()
    _, _, info = env.step(CloudAction(target_servers=1))
    assert info["sla_compliance"] == 1 - info["sla_violation"]
    assert info["overload"] >= 0


def test_reward_penalizes_sla_violation():
    config = load_config(mode="quick")
    ok = {"energy_wh": 10, "utilization": 0.7, "sla_violation": 0, "response_time_ms": 100, "overload": 0, "unmet_demand": 0, "churn": 0}
    bad = {**ok, "sla_violation": 1, "response_time_ms": 400, "overload": 0.5}
    assert compute_reward(bad, config) < compute_reward(ok, config)

