import numpy as np
from collections import defaultdict
from config import ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY

class SelfOptimizingAgent:
    """
    Q-learning agent for multi-agent resource management.
    
    State discretization uses workload demand, CPU statistics, capacity utilization,
    and time-of-day patterns to drive adaptive decisions.
    """
    
    def __init__(self, actions, seed=42, name="agent"):
        self.actions = actions  # Actions for this agent (e.g., [-1, 0, 1] or [0, 1])
        self.rng = np.random.default_rng(seed)
        self.q = defaultdict(lambda: np.zeros(len(actions), dtype=np.float32))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.name = name
        self.visit_count = defaultdict(int)
        self.episode_steps = 0
        
    def _discretize_state(self, s):
        """
        Convert continuous state to discrete state key.
        
        State features:
        - demand: workload demand (0-1)
        - mean_cpu: average CPU utilization (0-1)
        - p95_cpu: 95th percentile CPU utilization (0-1)
        - active_capacity: current active capacity fraction (0-1)
        - time_fraction: time of day (0-1)
        - trend: workload trend (increase/stable/decrease)
        """
        
        # Quantize the workload and control features used by the policy.
        demand_bin = min(9, int(np.clip(s.get("demand", 0) * 10, 0, 9)))
        previous_bin = min(9, int(np.clip(s.get("previous_demand", 0) * 10, 0, 9)))
        average_bin = min(9, int(np.clip(s.get("moving_average_demand", 0) * 10, 0, 9)))
        capacity_bin = min(9, int(np.clip(s.get("active_capacity", 0) * 10, 0, 9)))
        pressure_bin = min(9, int(np.clip(s.get("sla_pressure", 0) * 10, 0, 9)))
        volatility_bin = min(4, int(np.clip(s.get("demand_volatility", 0) * 20, 0, 4)))
        time_bin = min(11, int((s.get("time_fraction", 0) * 12) % 12))  # 12 buckets for 12 2-hour periods
        
        # Include trend if available
        trend = s.get("trend", 0)
        
        return (demand_bin, previous_bin, average_bin, capacity_bin,
            pressure_bin, volatility_bin, time_bin, trend)

    def act(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        """
        k = self._discretize_state(state)
        self.visit_count[k] += 1
        
        if training and self.rng.random() < self.epsilon:
            # Exploration: random action
            return int(self.rng.integers(len(self.actions)))
        else:
            # Exploitation: best action
            return int(np.argmax(self.q[k]))

    def learn(self, state, action, reward, next_state, done):
        """
        Q-learning update rule.
        """
        k = self._discretize_state(state)
        nk = self._discretize_state(next_state)
        
        # Compute target
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q[nk])
        
        # Update Q-value
        td_error = target - self.q[k][action]
        self.q[k][action] += self.alpha * td_error

    def decay_epsilon(self):
        """
        Decay exploration rate over episodes.
        """
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def get_q_stats(self):
        """Return Q-table statistics for analysis."""
        if not self.q:
            return {"states_visited": 0, "q_mean": 0, "q_std": 0, "q_max": 0}
        
        all_q = np.concatenate([v for v in self.q.values()])
        return {
            "states_visited": len(self.q),
            "q_mean": float(np.mean(all_q)),
            "q_std": float(np.std(all_q)),
            "q_max": float(np.max(all_q)),
        }


class ResourceAllocationAgent(SelfOptimizingAgent):
    """
    Agent responsible for scaling compute resources up/down.
    
    Actions:
    - -1: reduce target capacity (consolidate, scale down)
    -  0: hold target capacity
    - +1: increase target capacity (scale up)
    """
    def __init__(self, seed=42):
        super().__init__([-1, 0, 1], seed=seed, name="ResourceAgent")


class EnergyOptimizationAgent(SelfOptimizingAgent):
    """
    Agent responsible for energy-aware consolidation.
    
    Actions:
    - 0: hold (normal operation)
    - 1: consolidate (aggressive, move workloads to fewer servers)
    """
    def __init__(self, seed=42):
        super().__init__([0, 1], seed=seed, name="EnergyAgent")
