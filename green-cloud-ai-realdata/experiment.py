import numpy as np
import pandas as pd
from environment import TraceCloudEnvironment
from agents import ResourceAllocationAgent, EnergyOptimizationAgent
from config import (
    N_SERVERS, TRAIN_EPISODES, ALPHA, GAMMA, EPSILON_START, EPSILON_END,
    EPSILON_DECAY, ENERGY_WEIGHT, SLA_WEIGHT, RESPONSE_WEIGHT,
    UTILIZATION_WEIGHT, OVERLOAD_WEIGHT, CHURN_WEIGHT, WARNING_UTILIZATION_MAX,
)

REWARD_PRESETS = {
    "Energy-focused": (1.5, 5.0, 0.8, 0.5),
    "Balanced": (ENERGY_WEIGHT, SLA_WEIGHT, RESPONSE_WEIGHT, UTILIZATION_WEIGHT),
    "SLA-focused": (0.6, 14.0, 2.5, 0.4),
}


def compute_reward(info, weights=None):
    energy, sla, response, utilization = weights or (
        ENERGY_WEIGHT, SLA_WEIGHT, RESPONSE_WEIGHT, UTILIZATION_WEIGHT)
    efficient_reward = utilization * (1.0 if utilization <= WARNING_UTILIZATION_MAX else 0.25)
    overload_penalty = OVERLOAD_WEIGHT * info["overload"] ** 2
    return (-energy * info["energy_wh"] / 5000.0
            + efficient_reward * utilization
            - sla * info["sla_violation"]
            - response * max(0.0, info["response_ms"] - 200.0) / 100.0
            - overload_penalty
            - CHURN_WEIGHT * info["churn"] / N_SERVERS)


def train(trace, variant="full", seed=42, reward_weights=None):
    resource = ResourceAllocationAgent(seed=seed) if variant in ("full", "resource", "no_trend", "no_energy") else None
    energy = EnergyOptimizationAgent(seed=seed + 1) if variant in ("full", "energy", "no_trend", "no_energy") else None
    if variant == "no_energy":
        reward_weights = (0.0, SLA_WEIGHT, RESPONSE_WEIGHT, UTILIZATION_WEIGHT)
    history, diagnostics = [], []
    for episode in range(TRAIN_EPISODES):
        env = TraceCloudEnvironment(trace, seed=seed + 1000 + episode, use_trend=variant != "no_trend")
        state = env.reset()
        capacity, total_reward, actions = max(2, int(0.55 * N_SERVERS)), 0.0, []
        for _ in range(len(trace)):
            resource_action = resource.act(state, True) if resource else 1
            energy_action = energy.act(state, True) if energy else 0
            capacity = int(np.clip(capacity + (resource.actions[resource_action] if resource else 0), 1, N_SERVERS))
            next_state, done, info = env.step(capacity, bool(energy and energy.actions[energy_action]))
            reward = compute_reward(info, reward_weights)
            if resource: resource.learn(state, resource_action, reward, next_state, done)
            if energy: energy.learn(state, energy_action, reward, next_state, done)
            total_reward += reward
            actions.append((resource_action, energy_action))
            state = next_state
            if done: break
        if resource: resource.decay_epsilon()
        if energy: energy.decay_epsilon()
        q_stats = resource.get_q_stats() if resource else energy.get_q_stats()
        unique_actions = len(set(actions))
        history.append({"episode": episode + 1, "reward": total_reward,
                        "epsilon": resource.epsilon if resource else energy.epsilon,
                        "steps": len(actions), "unique_actions": unique_actions,
                        "learned_states": q_stats["states_visited"]})
        diagnostics.append({"episode": episode + 1, "resource_states": len(resource.q) if resource else 0,
                            "energy_states": len(energy.q) if energy else 0,
                            "resource_q_std": resource.get_q_stats()["q_std"] if resource else 0.0,
                            "energy_q_std": energy.get_q_stats()["q_std"] if energy else 0.0,
                            "unique_actions": unique_actions})
    return resource, energy, pd.DataFrame(history), pd.DataFrame(diagnostics)


def _target_from_policy(state, resource, energy, capacity, variant):
    resource_change = resource.actions[resource.act(state, False)] if resource else 0
    capacity = int(np.clip(capacity + resource_change, 1, N_SERVERS))
    consolidate = bool(energy and energy.actions[energy.act(state, False)] == 1)
    return capacity, consolidate


def evaluate(trace, resource_agent, energy_agent, seed, method, variant="full", reward_weights=None):
    env = TraceCloudEnvironment(trace, seed=seed, use_trend=variant != "no_trend")
    state, capacity, rows = env.reset(), max(2, int(0.55 * N_SERVERS)), []
    previous_demand = None
    for _ in range(len(trace)):
        demand = state["demand"]
        if method == "AI":
            capacity, consolidate = _target_from_policy(state, resource_agent, energy_agent, capacity, variant)
        elif method == "StaticOver":
            capacity, consolidate = int(0.75 * N_SERVERS), False
        elif method == "Threshold":
            if demand > 0.75 and capacity < N_SERVERS: capacity += 3
            elif demand < 0.35 and capacity > 1: capacity -= 3
            consolidate = False
        elif method == "Packing":
            capacity, consolidate = int(np.clip(np.ceil(demand * N_SERVERS * 1.10), 1, N_SERVERS)), True
        elif method == "Greedy":
            capacity, consolidate = int(np.clip(np.ceil(demand * N_SERVERS / 0.70), 1, N_SERVERS)), False
        else:
            raise ValueError(f"Unknown method: {method}")
        state, done, info = env.step(capacity, consolidate)
        info.update({"method": method, "seed": seed, "variant": variant})
        rows.append(info)
        previous_demand = demand
        if done: break
    return pd.DataFrame(rows)


def summarize_results(df):
    interval_minutes = 5.0
    return {
        "method": df["method"].iloc[0], "seed": int(df["seed"].iloc[0]),
        "energy_wh_total": df.energy_wh.sum(), "energy_wh_mean": df.energy_wh.mean(),
        "energy_per_workload": df.energy_wh.sum() / max(df.demand.sum(), 1e-9),
        "sla_violations": int(df.sla_violation.sum()),
        "sla_violation_rate": df.sla_violation.mean(), "sla_compliance_rate": df.sla_compliance.mean(),
        "mean_response_ms": df.response_ms.mean(), "p95_response_ms": df.response_ms.quantile(0.95),
        "mean_utilization": df.utilization.mean(), "p95_utilization": df.utilization.quantile(0.95),
        "mean_active_servers": df.active_servers.mean(), "peak_active_servers": df.active_servers.max(),
        "server_activation_count": int(df.server_activation.sum()),
        "server_deactivation_count": int(df.server_deactivation.sum()),
        "vm_migrations": int(df.vm_migrations.sum()), "total_churn": int(df.churn.sum()),
        "mean_overload": df.overload.mean(), "max_overload": df.overload.max(),
        "duration_hours": len(df) * interval_minutes / 60.0,
    }
