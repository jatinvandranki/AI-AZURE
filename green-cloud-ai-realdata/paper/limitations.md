## Limitations

The Azure trace exposes CPU utilization, not direct data-center electricity or end-user latency. Energy is estimated from a transparent server power model. The simulator uses homogeneous servers, modeled queueing/latency, and simplified migration costs. External validity depends on whether selected shards represent the target deployment. The Q-learning agents use discretized state features and may not generalize to production control planes without further validation. Training cost grows with trace length, candidate configurations, and ablation runs.
