# Paper methodology — trace-driven version

## Dataset
Microsoft Azure Public Dataset V2 (2019), a representative first-party Azure VM workload trace from one geographical region. The complete trace spans 30 consecutive days and contains approximately 2.7 million VMs and 5-minute CPU utilization readings.

## Trace processing
CPU readings are joined to the Azure VM table using the anonymized VM identifier. CPU demand is computed as:

CPU demand = VM virtual-core allocation × normalized CPU utilization.

Readings are aggregated into 5-minute intervals. The experiment uses a configurable subset of the CPU-reading files to make the study reproducible on ordinary hardware. The exact number of files and resulting rows must be reported.

## Environment
A trace-driven cloud simulator maps normalized workload demand onto a research testbed of 100 homogeneous physical servers. A best-fit/packing interpretation is used for the deterministic packing baseline.

## Energy model
The simulator estimates server power using:

P(u) = P_idle + (P_max - P_idle)u

where u is server utilization. Energy for each five-minute interval is:

E = P × (5/60) hours.

This is an analytical estimate, not a direct Azure power measurement.

## AI agents
Two cooperating Q-learning agents are used:
1. Resource agent: scale capacity down, hold, or scale capacity up.
2. Energy agent: hold or trigger consolidation.

Each agent observes workload demand, CPU statistics, active-capacity fraction and time-of-day information. The reward jointly penalizes energy, SLA violations, excessive response time and capacity churn, while rewarding useful utilization.

## Evaluation
The first 70% of the processed trace is used for training and the final 30% is held out for testing. The held-out period is evaluated across multiple random seeds.

Baselines:
- Static overprovisioning
- Threshold-based scaling
- Deterministic packing

Metrics:
- estimated energy
- SLA violations
- mean response time
- utilization
- active servers
- overload

## Scientific reporting
Do not claim that the AI "measured" Azure data-center energy. Say "estimated energy under the stated server power model." Also report the exact trace subset, preprocessing procedure, model parameters, seeds, and limitations.
