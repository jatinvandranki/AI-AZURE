from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .preprocessing import PreprocessingStats, add_workload_features, clean_cpu_chunk


CPU_FILE_RE = re.compile(r"file-(\d+)-of-\d+")


@dataclass
class DatasetReport:
    files_discovered: int = 0
    files_used: int = 0
    files_used_names: str = ""
    raw_rows: int = 0
    valid_rows: int = 0
    discarded_rows: int = 0
    unique_vm_identifiers: int = 0
    time_start: float | None = None
    time_end: float | None = None
    workload_intervals: int = 0
    missing_values: int = 0
    duplicate_observations: int = 0
    invalid_cpu_values: int = 0
    malformed_records: int = 0
    cpu_util_min: float | None = None
    cpu_util_max: float | None = None
    cpu_util_mean: float | None = None
    cpu_util_median: float | None = None
    temporal_coverage_hours: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def discover_trace_files(data_dir: Path) -> list[Path]:
    files = sorted([*data_dir.glob("*.csv"), *data_dir.glob("*.csv.gz")], key=_trace_sort_key)
    return [p for p in files if "cpu_readings" in p.name or "vm_cpu" in p.name]


def _trace_sort_key(path: Path) -> tuple[int, str]:
    match = CPU_FILE_RE.search(path.name)
    return (int(match.group(1)) if match else 0, path.name)


def _read_vm_table(data_dir: Path) -> pd.DataFrame:
    candidates = sorted(data_dir.glob("*vmtable*.csv*"))
    if not candidates:
        return pd.DataFrame(columns=["vm_id", "cores"])
    compression = "gzip" if candidates[0].suffix == ".gz" else None
    vm = pd.read_csv(candidates[0], compression=compression, header=None, usecols=[0, 9])
    vm.columns = ["vm_id", "cores"]
    vm["cores"] = pd.to_numeric(vm["cores"], errors="coerce").fillna(1.0).clip(0.25, 128.0)
    return vm.drop_duplicates("vm_id")


def build_workload(config) -> tuple[pd.DataFrame, DatasetReport]:
    data_dir = config.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = discover_trace_files(data_dir)
    report = DatasetReport(files_discovered=len(files))
    max_files = int(config.mode_settings.get("max_files", 0))
    max_rows = int(config.mode_settings.get("max_rows", 0))
    selected = files[:max_files] if max_files > 0 else files
    if not selected:
        raise FileNotFoundError(f"No Azure CPU trace .csv or .csv.gz shards found in {data_dir}")

    vm = _read_vm_table(data_dir)
    core_by_vm = vm.set_index("vm_id")["cores"] if not vm.empty else None
    interval_seconds = int(config.get("interval_seconds"))
    aggregate: dict[int, dict[str, object]] = {}
    seen_rows = 0
    vm_seen: set[object] = set()
    cpu_sum = 0.0
    cpu_count = 0
    cpu_hist = np.zeros(101, dtype=np.int64)

    for file_number, path in enumerate(selected, start=1):
        print(f"Reading Azure CPU shard {file_number}/{len(selected)}: {path.name}", flush=True)
        compression = "gzip" if path.suffix == ".gz" else None
        for chunk in pd.read_csv(path, compression=compression, header=None, usecols=[0, 1, 4], chunksize=config.get("chunk_size")):
            if max_rows and seen_rows >= max_rows:
                break
            if max_rows:
                chunk = chunk.head(max_rows - seen_rows)
            seen_rows += len(chunk)
            cleaned, stats = clean_cpu_chunk(chunk)
            _merge_stats(report, stats)
            if cleaned.empty:
                continue
            cpu_pct = np.clip(np.rint(cleaned["cpu_util"].to_numpy(dtype=float) * 100.0), 0, 100).astype(int)
            cpu_hist += np.bincount(cpu_pct, minlength=101)
            chunk_cpu_sum = float(cleaned["cpu_util"].sum())
            cpu_sum += chunk_cpu_sum
            cpu_count += len(cleaned)
            chunk_min = float(cleaned["cpu_util"].min())
            chunk_max = float(cleaned["cpu_util"].max())
            report.cpu_util_min = chunk_min if report.cpu_util_min is None else min(report.cpu_util_min, chunk_min)
            report.cpu_util_max = chunk_max if report.cpu_util_max is None else max(report.cpu_util_max, chunk_max)
            vm_seen.update(cleaned["vm_id"].unique().tolist())
            if report.time_start is None:
                report.time_start = float(cleaned["timestamp"].min())
                report.time_end = float(cleaned["timestamp"].max())
            else:
                report.time_start = min(report.time_start, float(cleaned["timestamp"].min()))
                report.time_end = max(report.time_end or 0, float(cleaned["timestamp"].max()))
            if core_by_vm is not None:
                cleaned["cores"] = cleaned["vm_id"].map(core_by_vm).fillna(1.0)
            else:
                cleaned["cores"] = 1.0
            cleaned["step"] = (cleaned["timestamp"] // interval_seconds).astype(int)
            cleaned["cpu_demand"] = cleaned["cores"] * cleaned["cpu_util"]
            grouped = cleaned.groupby("step", sort=False).agg(
                cpu_demand=("cpu_demand", "sum"),
                cpu_sum=("cpu_util", "sum"),
                cpu_count=("cpu_util", "size"),
                max_cpu_util=("cpu_util", "max"),
                vm_count=("vm_id", "nunique"),
            )
            for step, row in grouped.iterrows():
                bucket = aggregate.setdefault(
                    int(step),
                    {"cpu_demand": 0.0, "cpu_sum": 0.0, "cpu_count": 0, "max_cpu_util": 0.0, "vm_count": 0},
                )
                bucket["cpu_demand"] = float(bucket["cpu_demand"]) + float(row["cpu_demand"])
                bucket["cpu_sum"] = float(bucket["cpu_sum"]) + float(row["cpu_sum"])
                bucket["cpu_count"] = int(bucket["cpu_count"]) + int(row["cpu_count"])
                bucket["max_cpu_util"] = max(float(bucket["max_cpu_util"]), float(row["max_cpu_util"]))
                bucket["vm_count"] = max(int(bucket["vm_count"]), int(row["vm_count"]))
        if max_rows and seen_rows >= max_rows:
            break

    rows = []
    for step, bucket in aggregate.items():
        cpu_count_step = int(bucket["cpu_count"])
        rows.append({
            "step": step,
            "cpu_demand": bucket["cpu_demand"],
            "vm_count": int(bucket["vm_count"]),
            "mean_cpu_util": float(bucket["cpu_sum"]) / max(cpu_count_step, 1),
            "p95_cpu_util": float(bucket["max_cpu_util"]),
            "max_cpu_util": float(bucket["max_cpu_util"]),
        })
    if not rows:
        raise ValueError("No valid workload intervals could be constructed from the selected traces")
    trace = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    trace.attrs["interval_seconds"] = interval_seconds
    trace = add_workload_features(trace)
    report.files_used = len(selected)
    report.files_used_names = ";".join(p.name for p in selected)
    report.unique_vm_identifiers = len(vm_seen)
    report.workload_intervals = len(trace)
    report.cpu_util_mean = cpu_sum / cpu_count if cpu_count else None
    report.cpu_util_median = _histogram_median(cpu_hist) if cpu_count else None
    if report.time_start is not None and report.time_end is not None:
        report.temporal_coverage_hours = (report.time_end - report.time_start) / 3600.0
    trace.attrs["dataset_report"] = report.to_dict()
    trace.attrs["files_used"] = [p.name for p in selected]
    return trace, report


def _histogram_median(hist: np.ndarray) -> float:
    total = int(hist.sum())
    if total == 0:
        return 0.0
    midpoint = (total - 1) // 2
    cumulative = np.cumsum(hist)
    lower = int(np.searchsorted(cumulative, midpoint + 1))
    if total % 2:
        return lower / 100.0
    upper = int(np.searchsorted(cumulative, midpoint + 2))
    return ((lower + upper) / 2.0) / 100.0


def _merge_stats(report: DatasetReport, stats: PreprocessingStats) -> None:
    report.raw_rows += stats.raw_rows
    report.valid_rows += stats.valid_rows
    report.discarded_rows += stats.discarded_rows
    report.missing_values += stats.missing_values
    report.duplicate_observations += stats.duplicate_observations
    report.invalid_cpu_values += stats.invalid_cpu_values
    report.malformed_records += stats.malformed_records
