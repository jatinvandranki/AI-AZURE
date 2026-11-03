import numpy as np
from config import (
    N_SERVERS, IDLE_POWER_W, MAX_POWER_W, TARGET_RESPONSE_MS,
    INTERVAL_MINUTES, RESPONSE_BASE_MS, RESPONSE_UTILIZATION_EXPONENT,
    QUEUE_PRESSURE_WEIGHT, OVERLOAD_RESPONSE_WEIGHT,
    EFFICIENT_UTILIZATION_MAX, WARNING_UTILIZATION_MAX,
    HIGH_RISK_UTILIZATION_MAX,
)


class TraceCloudEnvironment:
    """Trace-driven simulator with bounded capacity and an explicit SLA model."""

    def __init__(self, trace, seed=42, use_trend=True):
        self.trace = trace.reset_index(drop=True)
        self.rng = np.random.default_rng(seed)
        self.use_trend = use_trend
        self.trends = self._compute_trends(self.trace["normalized_demand"].to_numpy())
        self.reset()

    @staticmethod
    def _compute_trends(demands, window=5):
        trends = np.zeros(len(demands), dtype=int)
        for index in range(len(demands)):
            recent = demands[max(0, index - window):index]
            if len(recent) and demands[index] > np.mean(recent) * 1.1:
                trends[index] = 1
            elif len(recent) and demands[index] < np.mean(recent) * 0.9:
                trends[index] = -1
        return trends

    def reset(self, start_index=0):
        self.t = start_index
        self.previous_servers = max(2, int(0.55 * N_SERVERS))
        return self.state()

    def state(self):
        index = min(self.t, len(self.trace) - 1)
        row = self.trace.iloc[index]
        demand = float(row["normalized_demand"])
        active = self.previous_servers / N_SERVERS
        pressure = demand / max(active, 0.01)
        return {
            "demand": demand,
            "previous_demand": float(row.get("previous_demand", demand)),
            "moving_average_demand": float(row.get("moving_average_demand", demand)),
            "demand_volatility": float(row.get("demand_volatility", 0.0)),
            "mean_cpu": float(row["mean_cpu_util"]),
            "p95_cpu": float(row["p95_cpu_util"]),
            "active_capacity": active,
            "spare_capacity": max(0.0, 1.0 - pressure),
            "sla_pressure": max(0.0, pressure - EFFICIENT_UTILIZATION_MAX),
            "energy_estimate": active * (IDLE_POWER_W + (MAX_POWER_W - IDLE_POWER_W) * min(1.0, pressure)),
            "time_fraction": (self.t % 288) / 288.0,
            "trend": int(self.trends[index]) if self.use_trend else 0,
        }

    def step(self, target_capacity, consolidate=False):
        if self.t >= len(self.trace):
            raise RuntimeError("Environment episode already finished; call reset()")
        row = self.trace.iloc[self.t]
        demand = float(row["normalized_demand"])
        target_servers = int(np.clip(round(target_capacity), 1, N_SERVERS))
        if consolidate:
            target_servers = max(1, target_servers - 1)
        capacity_utilization = demand * N_SERVERS / target_servers
        overload = max(0.0, capacity_utilization - 1.0)
        utilization = min(1.0, capacity_utilization)
        power_per_server = IDLE_POWER_W + (MAX_POWER_W - IDLE_POWER_W) * utilization
        energy_wh = target_servers * power_per_server * (INTERVAL_MINUTES / 60.0)
        queue_pressure = max(0.0, utilization - WARNING_UTILIZATION_MAX)
        response_ms = RESPONSE_BASE_MS * (1.0 + RESPONSE_UTILIZATION_EXPONENT * utilization ** 2)
        response_ms += QUEUE_PRESSURE_WEIGHT * queue_pressure
        response_ms += OVERLOAD_RESPONSE_WEIGHT * overload ** 2
        response_ms = max(1.0, response_ms + float(self.rng.normal(0, 2.0)))
        sla_violation = int(response_ms > TARGET_RESPONSE_MS)
        churn = abs(target_servers - self.previous_servers)
        info = {
            "step": int(row.get("step", self.t)),
            "energy_wh": energy_wh,
            "response_ms": response_ms,
            "sla_violation": sla_violation,
            "sla_compliance": 1 - sla_violation,
            "utilization": utilization,
            "active_servers": target_servers,
            "vm_count": int(row["vm_count"]),
            "demand": demand,
            "overload": overload,
            "churn": churn,
            "server_activation": max(0, target_servers - self.previous_servers),
            "server_deactivation": max(0, self.previous_servers - target_servers),
            "vm_migrations": int(churn * 0.1) if consolidate else 0,
            "mean_cpu": float(row["mean_cpu_util"]),
            "p95_cpu": float(row["p95_cpu_util"]),
            "utilization_zone": "efficient" if utilization <= EFFICIENT_UTILIZATION_MAX else "warning" if utilization <= WARNING_UTILIZATION_MAX else "high-risk" if utilization <= HIGH_RISK_UTILIZATION_MAX else "overload",
        }
        self.previous_servers = target_servers
        self.t += 1
        return self.state(), self.t >= len(self.trace), info
