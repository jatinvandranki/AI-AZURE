"""
Configuration for Azure trace-driven cloud simulation experiment.
"Sustainable Green Cloud Computing Using Self-Optimizing AI Agents"
"""

SEED = 42

# === AZURE PUBLIC DATASET V2 ===
# CPU trace files to use (1 to 195 available)
# Note: Files are HEADERLESS with columns:
#   timestamp, vm_id, min_cpu_pct, max_cpu_pct, avg_cpu_pct
# CPU values are percentages (0-100), automatically normalized to [0,1]
# Each file contains ~50 days of 5-minute interval data
CPU_FILES = 0              # 0 = discover every locally available CPU shard
DATA_DIR = "data/azure_v2"

# === TIME AGGREGATION ===
INTERVAL_MINUTES = 5         # 5-minute aggregation intervals
STEPS_PER_DAY = 288          # 24*60/5 = 288 5-minute intervals per day
MAX_STEPS = 0                # 0 = keep every usable chronological interval
MIN_VMS_PER_STEP = 1         # Minimum VMs per interval to keep (filters empty periods)

# === SIMULATION ENVIRONMENT ===
# Simulated cloud environment with homogeneous servers
N_SERVERS = 100              # Total physical server capacity
SERVER_CPU = 100.0           # CPU cores per server
SERVER_RAM = 100.0           # Memory GB per server
IDLE_POWER_W = 120.0         # Idle power consumption (watts)
MAX_POWER_W = 250.0          # Maximum power consumption (watts)
# Server power model: P(u) = P_idle + (P_max - P_idle) * u
#   where u = server utilization in [0,1]

# === SLA & PERFORMANCE ===
TARGET_RESPONSE_MS = 200.0   # Target response time SLA (milliseconds)
EFFICIENT_UTILIZATION_MAX = 0.70
WARNING_UTILIZATION_MAX = 0.85
HIGH_RISK_UTILIZATION_MAX = 1.00
RESPONSE_BASE_MS = 80.0
RESPONSE_UTILIZATION_EXPONENT = 2.0
QUEUE_PRESSURE_WEIGHT = 120.0
OVERLOAD_RESPONSE_WEIGHT = 500.0
SERVER_TRANSITION_MINUTES = 5.0

# === Q-LEARNING HYPERPARAMETERS ===
# Resource Allocation Agent
ALPHA = 0.12                 # Learning rate
GAMMA = 0.94                 # Discount factor
EPSILON_START = 1.0          # Initial exploration probability
EPSILON_END = 0.05           # Final exploration probability
EPSILON_DECAY = 0.985        # Epsilon decay rate per episode

# === REWARD FUNCTION WEIGHTS ===
# Multi-objective reward: balance energy, SLA, performance, efficiency, churn
ENERGY_WEIGHT = 1.0          # Estimated energy penalty, normalized by interval
SLA_WEIGHT = 8.0             # Penalty for response_ms > TARGET_RESPONSE_MS
RESPONSE_WEIGHT = 1.5        # Penalty for latency above the SLA target
UTILIZATION_WEIGHT = 0.8     # Reward for utilization in the efficient zone
OVERLOAD_WEIGHT = 10.0       # Nonlinear penalty for utilization above 1.0
CHURN_WEIGHT = 0.10          # Penalty for server transitions

# === TRAINING CONFIGURATION ===
TRAIN_EPISODES = 80          # Number of training episodes
# Each episode runs through the entire training trace with random seed

# === TESTING CONFIGURATION ===
TEST_EPISODES = 12           # Deprecated (not used in new pipeline)
TEST_SEEDS = [101, 202, 303, 404, 505]
# Multiple seeds ensure statistical robustness of results
