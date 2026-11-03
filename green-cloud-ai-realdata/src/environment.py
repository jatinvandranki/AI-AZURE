from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CloudAction:
    target_servers: int
    consolidate: bool = False


class CloudEnvironment:
    def __init__(self, trace: pd.DataFrame, config, seed: int = 0, use_trend: bool = True):
        self.trace = trace.reset_index(drop=True)
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.use_trend = use_trend
        self.max_servers = int(config.get("max_servers"))
        self.server_capacity = float(config.get("server_capacity"))
        self.interval_hours = float(config.get("interval_seconds")) / 3600.0
        self.reset()

    def reset(self):
        self.t = 0
        self.previous_servers = int(self.config.get("initial_active_servers"))
        self.previous_response_ms = float(self.config.get("base_latency_ms"))
        self.previous_overload = 0.0
        return self.state()

    def state(self) -> dict[str, float | int | str]:
        row = self.trace.iloc[min(self.t, len(self.trace) - 1)]
        demand = float(row["normalized_demand"])
        active = self.previous_servers / self.max_servers
        capacity = self.previous_servers / self.max_servers
        pressure = demand / max(capacity, 1e-9)
        energy = self.estimate_energy(self.previous_servers, min(1.0, pressure))
        trend = row.get("trend", "stable") if self.use_trend else "stable"
        return {
            "demand": demand,
            "previous_demand": float(row.get("previous_demand", demand)),
            "moving_average_demand": float(row.get("moving_average_demand", demand)),
            "demand_volatility": float(row.get("demand_volatility", 0.0)),
            "mean_cpu": float(row.get("mean_cpu_util", 0.0)),
            "p95_cpu": float(row.get("p95_cpu_util", 0.0)),
            "active_servers": self.previous_servers,
            "active_capacity": active,
            "available_capacity": capacity,
            "spare_capacity": max(0.0, capacity - demand),
            "sla_pressure": max(0.0, pressure - self.config.get("utilization_regions")["efficient_max"]),
            "current_estimated_energy": energy,
            "recent_response_time": self.previous_response_ms,
            "recent_overload": self.previous_overload,
            "time_fraction": (int(row["step"]) % int(86400 / self.config.get("interval_seconds"))) / (86400 / self.config.get("interval_seconds")),
            "trend": trend,
        }

    def estimate_energy(self, active_servers: int, utilization: float, power_model: dict | None = None) -> float:
        power = power_model or self.config.power
        u = float(np.clip(utilization, 0.0, 1.0))
        watts = power["p_idle"] + (power["p_max"] - power["p_idle"]) * u
        return active_servers * watts * self.interval_hours

    def latency_ms(self, utilization: float, overload: float) -> float:
        base = float(self.config.get("base_latency_ms"))
        regions = self.config.get("utilization_regions")
        queue_pressure = max(0.0, utilization - regions["normal_max"])
        latency = base * (1.0 + 2.0 * utilization * utilization)
        latency += float(self.config.get("queue_delay_weight")) * (queue_pressure ** 2)
        latency += float(self.config.get("overload_latency_weight")) * (overload ** 2)
        return max(base, latency)

    def step(self, action: CloudAction) -> tuple[dict, bool, dict]:
        row = self.trace.iloc[self.t]
        target = int(np.clip(action.target_servers, 1, self.max_servers))
        if action.consolidate:
            required = int(np.ceil(float(row["normalized_demand"]) * self.max_servers / 0.95))
            target = max(1, max(required, target - 1))
        target = int(np.clip(target, 1, self.max_servers))
        demand = float(row["normalized_demand"])
        utilization_raw = demand * self.max_servers / target
        overload = max(0.0, utilization_raw - 1.0)
        utilization = min(1.0, utilization_raw)
        energy_wh = self.estimate_energy(target, utilization)
        response_ms = self.latency_ms(utilization, overload)
        unmet_demand = max(0.0, demand - target / self.max_servers)
        sla_violation = int(
            response_ms > float(self.config.get("sla_threshold_ms")) or overload > 0.0 or unmet_demand > 0.0
        )
        activations = max(0, target - self.previous_servers)
        deactivations = max(0, self.previous_servers - target)
        migrations = int(np.ceil(deactivations * 0.05)) if action.consolidate and deactivations else 0
        info = {
            "interval": self.t,
            "step": int(row["step"]),
            "timestamp": float(row.get("timestamp", row["step"] * self.config.get("interval_seconds"))),
            "workload": demand,
            "energy_wh": energy_wh,
            "sla_violation": sla_violation,
            "sla_compliance": 1 - sla_violation,
            "response_time_ms": response_ms,
            "utilization": utilization,
            "active_servers": target,
            "overload": overload,
            "unmet_demand": unmet_demand,
            "churn": activations + deactivations + migrations,
            "server_activation_count": activations,
            "server_deactivation_count": deactivations,
            "vm_migration_count": migrations,
            "consolidation_count": int(action.consolidate),
        }
        self.previous_servers = target
        self.previous_response_ms = response_ms
        self.previous_overload = overload
        self.t += 1
        return self.state(), self.t >= len(self.trace), info


def compute_reward(info: dict, config, reward_override: dict | None = None) -> float:
    weights = dict(config.get("reward"))
    if reward_override:
        weights.update(reward_override)
    utilization = info["utilization"]
    regions = config.get("utilization_regions")
    utilization_reward = utilization if regions["efficient_max"] <= utilization <= regions["warning_max"] else 0.25 * utilization
    latency_excess = max(0.0, info["response_time_ms"] - config.get("sla_threshold_ms")) / config.get("sla_threshold_ms")
    return (
        -weights["energy"] * info["energy_wh"] / 1000.0
        + weights["utilization"] * utilization_reward
        - weights["sla"] * info["sla_violation"]
        - weights["latency"] * latency_excess
        - weights["overload"] * (info["overload"] ** 2 + info["unmet_demand"] ** 2)
        - weights["churn"] * info["churn"] / config.get("max_servers")
    )
