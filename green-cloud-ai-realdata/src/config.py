from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "data_dir": "data/azure_v2",
    "modes": {
        "quick": {"max_files": 1, "max_rows": 2000000, "num_episodes": 8, "min_intervals": 8, "strict_publication": False},
        "standard": {"max_files": 3, "max_rows": 3000000, "num_episodes": 40, "min_intervals": 60, "strict_publication": False},
        "full": {"max_files": 0, "max_rows": 0, "num_episodes": 120, "min_intervals": 2000, "strict_publication": True},
    },
    "dataset": {
        "release_base_url": "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2/",
        "vm_table": "trace_data_vmtable_vmtable.csv.gz",
        "cpu_file_pattern": "trace_data_vm_cpu_readings_vm_cpu_readings-file-{index}-of-195.csv.gz",
        "total_cpu_files": 195,
        "publication_target_intervals": 2000,
    },
    "interval_seconds": 300,
    "train_ratio": 0.60,
    "validation_ratio": 0.20,
    "test_ratio": 0.20,
    "server_capacity": 100.0,
    "max_servers": 100,
    "initial_active_servers": 55,
    "power_models": {
        "conservative": {"p_idle": 80.0, "p_max": 180.0},
        "nominal": {"p_idle": 120.0, "p_max": 250.0},
        "high_power": {"p_idle": 180.0, "p_max": 400.0},
    },
    "power_model": "nominal",
    "base_latency_ms": 80.0,
    "sla_threshold_ms": 200.0,
    "queue_delay_weight": 180.0,
    "overload_latency_weight": 900.0,
    "utilization_regions": {"efficient_max": 0.70, "normal_max": 0.85, "warning_max": 0.95, "high_risk_max": 1.00},
    "learning_rate": 0.12,
    "discount_factor": 0.94,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.985,
    "reward": {"energy": 1.0, "sla": 8.0, "latency": 1.5, "overload": 12.0, "churn": 0.15, "utilization": 0.8},
    "seeds": [101, 202, 303, 404, 505],
    "chunk_size": 1000000,
    "bootstrap_samples": 1000,
}


@dataclass(frozen=True)
class ExperimentConfig:
    values: dict[str, Any]
    mode: str
    root: Path

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    @property
    def data_dir(self) -> Path:
        p = Path(self.values["data_dir"])
        return p if p.is_absolute() else self.root / p

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def plots_dir(self) -> Path:
        return self.root / "figures"

    @property
    def paper_dir(self) -> Path:
        return self.root / "paper"

    @property
    def mode_settings(self) -> dict[str, Any]:
        return self.values["modes"][self.mode]

    @property
    def power(self) -> dict[str, float]:
        return self.values["power_models"][self.values["power_model"]]


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path = "config/config.yaml", mode: str = "quick", **overrides: Any) -> ExperimentConfig:
    root = Path(__file__).resolve().parents[1]
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    if yaml is None:
        values = deepcopy(DEFAULT_CONFIG)
    else:
        with config_path.open("r", encoding="utf-8") as fh:
            values = yaml.safe_load(fh)
    if mode not in values["modes"]:
        raise ValueError(f"Unknown mode {mode!r}; choose one of {sorted(values['modes'])}")
    values = deep_update(values, {k: v for k, v in overrides.items() if v is not None})
    return ExperimentConfig(values=values, mode=mode, root=root)
