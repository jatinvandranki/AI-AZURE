from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path


CPU_RE = re.compile(r"trace_data_vm_cpu_readings_vm_cpu_readings-file-(\d+)-of-195\.csv(?:\.gz)?$")


@dataclass
class DatasetRequirement:
    data_dir: str
    vm_table_name: str
    vm_table_present: bool
    cpu_files_present: int
    cpu_file_indices_present: list[int]
    current_intervals: int
    target_intervals: int
    estimated_intervals_per_file: float
    required_cpu_files_total: int
    additional_cpu_files_needed: int
    next_cpu_file_indices: list[int]
    approximate_existing_cpu_size_gb: float
    approximate_required_download_gb: float
    approximate_final_dataset_gb: float
    sufficient: bool

    def to_dict(self) -> dict:
        return asdict(self)


def inventory_data_dir(config, current_intervals: int | None = None) -> DatasetRequirement:
    data_dir = config.data_dir
    dataset = config.get("dataset")
    target = int(dataset.get("publication_target_intervals", config.mode_settings.get("min_intervals", 2000)))
    vm_table = data_dir / dataset["vm_table"]
    cpu_files = []
    for path in sorted(data_dir.glob("trace_data_vm_cpu_readings_vm_cpu_readings-file-*-of-195.csv*")):
        match = CPU_RE.match(path.name)
        if match:
            cpu_files.append((int(match.group(1)), path))
    indices = sorted(i for i, _ in cpu_files)
    existing_size = sum(p.stat().st_size for _, p in cpu_files) / (1024 ** 3)
    observed_intervals = int(current_intervals or 0)
    intervals_per_file = observed_intervals / len(cpu_files) if cpu_files and observed_intervals else 45.0
    required_total = min(int(dataset["total_cpu_files"]), max(1, math.ceil(target / max(intervals_per_file, 1e-9))))
    missing_indices = [i for i in range(1, required_total + 1) if i not in set(indices)]
    avg_cpu_size = existing_size / len(cpu_files) if cpu_files else 0.85
    vm_size = vm_table.stat().st_size / (1024 ** 3) if vm_table.exists() else 0.41
    required_download = len(missing_indices) * avg_cpu_size + (0 if vm_table.exists() else vm_size)
    final_size = existing_size + required_download + (vm_size if vm_table.exists() else 0)
    return DatasetRequirement(
        data_dir=str(data_dir),
        vm_table_name=dataset["vm_table"],
        vm_table_present=vm_table.exists(),
        cpu_files_present=len(cpu_files),
        cpu_file_indices_present=indices,
        current_intervals=observed_intervals,
        target_intervals=target,
        estimated_intervals_per_file=intervals_per_file,
        required_cpu_files_total=required_total,
        additional_cpu_files_needed=len(missing_indices),
        next_cpu_file_indices=missing_indices,
        approximate_existing_cpu_size_gb=round(existing_size, 2),
        approximate_required_download_gb=round(required_download, 2),
        approximate_final_dataset_gb=round(final_size, 2),
        sufficient=observed_intervals >= target and vm_table.exists(),
    )


def required_file_names(config, requirement: DatasetRequirement) -> list[str]:
    dataset = config.get("dataset")
    names = []
    if not requirement.vm_table_present:
        names.append(dataset["vm_table"])
    names.extend(dataset["cpu_file_pattern"].format(index=i) for i in requirement.next_cpu_file_indices)
    return names

