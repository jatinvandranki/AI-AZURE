from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


TREND_MAP = {"falling": -1, "stable": 0, "rising": 1, "volatile": 2}


class QLearningAgent:
    def __init__(self, actions: list[int], config, seed: int, name: str, features: list[str]):
        self.actions = actions
        self.config = config
        self.name = name
        self.features = features
        self.rng = np.random.default_rng(seed)
        self.q = defaultdict(lambda: np.zeros(len(actions), dtype=float))
        self.epsilon = float(config.get("epsilon_start"))
        self.action_counts = Counter()
        self.last_policy: dict[tuple, int] = {}

    def discretize(self, state: dict) -> tuple:
        bins = []
        for feature in self.features:
            value = state.get(feature, 0)
            if feature == "trend":
                bins.append(TREND_MAP.get(value, 0))
            elif feature == "active_servers":
                bins.append(int(np.clip(value, 0, self.config.get("max_servers")) // 5))
            else:
                bins.append(int(np.clip(float(value), 0.0, 1.5) * 10))
        return tuple(bins)

    def act(self, state: dict, training: bool) -> int:
        key = self.discretize(state)
        if training and self.rng.random() < self.epsilon:
            idx = int(self.rng.integers(len(self.actions)))
        else:
            idx = int(np.argmax(self.q[key]))
        self.action_counts[self.actions[idx]] += 1
        return idx

    def learn(self, state: dict, action_idx: int, reward: float, next_state: dict, done: bool) -> None:
        key = self.discretize(state)
        next_key = self.discretize(next_state)
        target = reward if done else reward + self.config.get("discount_factor") * float(np.max(self.q[next_key]))
        self.q[key][action_idx] += self.config.get("learning_rate") * (target - self.q[key][action_idx])

    def end_episode(self) -> float:
        stable = 0
        total = 0
        policy = {state: int(np.argmax(values)) for state, values in self.q.items()}
        for state, action in policy.items():
            if state in self.last_policy:
                total += 1
                stable += int(self.last_policy[state] == action)
        self.last_policy = policy
        self.epsilon = max(float(self.config.get("epsilon_min")), self.epsilon * float(self.config.get("epsilon_decay")))
        return stable / total if total else np.nan

    def stats(self) -> dict:
        q_values = np.concatenate(list(self.q.values())) if self.q else np.array([0.0])
        return {
            "states": len(self.q),
            "q_table_size": int(sum(len(v) for v in self.q.values())),
            "q_mean": float(np.mean(q_values)),
            "q_std": float(np.std(q_values)),
            "q_max": float(np.max(q_values)),
            "epsilon": self.epsilon,
        }


def make_agents(config, seed: int, variant: str = "full"):
    resource = None
    energy = None
    if variant in {"full", "no_trend", "no_energy", "resource_only", "no_coordination"}:
        resource = QLearningAgent(
            actions=[-2, -1, 0, 1, 2],
            config=config,
            seed=seed,
            name="ResourceAllocationAgent",
            features=["demand", "previous_demand", "moving_average_demand", "active_servers", "sla_pressure", "trend"],
        )
    if variant in {"full", "no_trend", "no_energy", "energy_only", "no_coordination"}:
        energy = QLearningAgent(
            actions=[0, 1],
            config=config,
            seed=seed + 17,
            name="EnergyOptimizationAgent",
            features=["demand", "demand_volatility", "spare_capacity", "recent_response_time", "recent_overload", "trend"],
        )
    return resource, energy

