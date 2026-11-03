from __future__ import annotations

import numpy as np

from .environment import CloudAction


def baseline_action(method: str, state: dict, previous_servers: int, config) -> CloudAction:
    max_servers = int(config.get("max_servers"))
    demand = float(state["demand"])
    required_70 = int(np.ceil(demand * max_servers / 0.70))
    required_85 = int(np.ceil(demand * max_servers / 0.85))
    if method == "StaticOver":
        return CloudAction(target_servers=max(required_70, int(0.80 * max_servers)), consolidate=False)
    if method == "Threshold":
        target = previous_servers
        pressure = demand / max(previous_servers / max_servers, 1e-9)
        if pressure > 0.82:
            target += 4
        elif pressure < 0.50:
            target -= 2
        return CloudAction(target_servers=int(np.clip(target, required_85, max_servers)), consolidate=False)
    if method == "Packing":
        return CloudAction(target_servers=int(np.clip(required_85, 1, max_servers)), consolidate=True)
    if method == "Greedy":
        target = int(np.clip(required_70, 1, max_servers))
        return CloudAction(target_servers=target, consolidate=target < previous_servers)
    raise ValueError(f"Unknown baseline method: {method}")

