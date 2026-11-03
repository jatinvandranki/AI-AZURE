## Methodology

The experiment uses Microsoft Azure Public Dataset V2 CPU readings. CPU values are interpreted as percentages and divided by 100. Workload intervals are constructed chronologically with no random shuffling before the 60/20/20 train/validation/test split. All methods use the same homogeneous-server simulator, linear power model, latency model, and SLA definition.

Energy is computed as P(u) = P_idle + (P_max - P_idle) * u, integrated over each interval. This is a model-based estimate of energy consumption and not direct measurement of Azure data-center energy.
