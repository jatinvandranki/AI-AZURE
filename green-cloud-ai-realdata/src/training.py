from __future__ import annotations

import numpy as np
import pandas as pd

from .agents import make_agents
from .environment import CloudAction, CloudEnvironment, compute_reward


REWARD_PRESETS = {
    "Energy-focused": {"energy": 1.6, "sla": 6.0, "latency": 1.0},
    "Balanced": {},
    "SLA-focused": {"energy": 0.7, "sla": 14.0, "latency": 2.5},
}


def coordinated_action(resource, energy, state: dict, capacity: int, config, variant: str, training: bool = True) -> tuple[CloudAction, tuple[int | None, int | None]]:
    resource_idx = resource.act(state, training) if resource else None
    energy_idx = energy.act(state, training) if energy else None
    delta = resource.actions[resource_idx] if resource else 0
    consolidate = bool(energy and energy.actions[energy_idx] == 1)
    if variant != "no_coordination" and consolidate and state["sla_pressure"] > 0.05:
        consolidate = False
    target = int(np.clip(capacity + delta, 1, config.get("max_servers")))
    if state["trend"] == "rising" and variant != "no_trend":
        target += 1
    return CloudAction(target_servers=target, consolidate=consolidate), (resource_idx, energy_idx)


def train_policy(trace: pd.DataFrame, config, seed: int, variant: str = "full", reward_override: dict | None = None):
    resource, energy = make_agents(config, seed, variant)
    if variant == "no_energy":
        reward_override = {**(reward_override or {}), "energy": 0.0}
    history = []
    diagnostics = []
    episodes = int(config.mode_settings.get("num_episodes"))
    for episode in range(episodes):
        env = CloudEnvironment(trace, config, seed=seed + episode, use_trend=variant != "no_trend")
        state = env.reset()
        capacity = int(config.get("initial_active_servers"))
        total_reward = 0.0
        action_labels = []
        for _ in range(len(trace)):
            action, indices = coordinated_action(resource, energy, state, capacity, config, variant, training=True)
            capacity = action.target_servers
            next_state, done, info = env.step(action)
            reward = compute_reward(info, config, reward_override)
            if resource and indices[0] is not None:
                resource.learn(state, indices[0], reward, next_state, done)
            if energy and indices[1] is not None:
                energy.learn(state, indices[1], reward, next_state, done)
            total_reward += reward
            action_labels.append(str((action.target_servers, action.consolidate)))
            state = next_state
            if done:
                break
        stability_values = [a.end_episode() for a in (resource, energy) if a is not None]
        finite_stability = [x for x in stability_values if not np.isnan(x)]
        stability = float(np.mean(finite_stability)) if finite_stability else np.nan
        rstats = resource.stats() if resource else {"states": 0, "q_table_size": 0, "q_std": 0.0, "epsilon": energy.epsilon}
        estats = energy.stats() if energy else {"states": 0, "q_table_size": 0, "q_std": 0.0}
        history.append({
            "episode": episode + 1,
            "reward": total_reward,
            "epsilon": rstats["epsilon"],
            "q_table_size": rstats["q_table_size"] + estats["q_table_size"],
            "visited_states": rstats["states"] + estats["states"],
            "unique_actions": len(set(action_labels)),
            "policy_stability": stability,
        })
        diagnostics.append({
            "episode": episode + 1,
            "resource_states": rstats["states"],
            "energy_states": estats["states"],
            "resource_q_std": rstats["q_std"],
            "energy_q_std": estats["q_std"],
            "policy_stability": stability,
        })
    return resource, energy, pd.DataFrame(history), pd.DataFrame(diagnostics)
