from pathlib import Path
import gzip
import pandas as pd
import numpy as np
from config import DATA_DIR, CPU_FILES, INTERVAL_MINUTES, MAX_STEPS, MIN_VMS_PER_STEP

DATA = Path(DATA_DIR)

def load_vm_table():
    """
    Load Azure VM table (headerless).
    
    Azure VM table columns (0-indexed):
    0: vm_id
    1: subscription_id  
    2: deployment_id
    3: time_stamp (start)
    4: time_stamp (end)
    5: cpu_core_count (average)
    6: memory (GB average)
    7: memory (maximum)
    8: vm_category
    9: vm_cpu_core (actual bucket)
    10: vm_memory (actual bucket)
    """
    p = DATA / "trace_data_vmtable_vmtable.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"VM table not found at {p}. Run setup_azure_data.py first.")
    
    # Read without header
    df = pd.read_csv(p, compression="gzip", header=None)
    print(f"VM table loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Azure V2 schema: assign standard column names
    if len(df.columns) >= 10:
        df = df.iloc[:, [0, 9, 10]].copy()  # vm_id, cpu_cores, memory
        df.columns = ["vm_id", "cores", "memory"]
    else:
        raise ValueError(f"VM table has unexpected structure: {len(df.columns)} columns")
    
    df["cores"] = pd.to_numeric(df["cores"], errors="coerce").fillna(2.0)
    df["memory"] = pd.to_numeric(df["memory"], errors="coerce").fillna(4.0)
    df["cores"] = df["cores"].clip(lower=0.25, upper=128.0)
    df["memory"] = df["memory"].clip(lower=0.5, upper=1024.0)
    
    return df.drop_duplicates("vm_id")


def build_trace():
    """
    Build trace from Azure Public Dataset V2.
    
    CPU trace is headerless with columns (0-indexed):
    0: timestamp (seconds since epoch)
    1: vm_id
    2: min_cpu_utilization (percentage, 0-100)
    3: max_cpu_utilization (percentage, 0-100)
    4: avg_cpu_utilization (percentage, 0-100)
    
    The key difference from synthetic traces:
    - CPU values are PERCENTAGES (0-100), not normalized (0-1)
    - Must divide by 100 to get utilization
    - Timestamps are in seconds
    - Data is aggregated into 5-minute intervals
    """
    
    print("\n=== AZURE TRACE PROCESSING ===")
    vm = load_vm_table()
    print(f"Loaded VM metadata: {len(vm)} unique VMs")
    
    chunks = []
    total_rows_read = 0

    available = sorted(DATA.glob("trace_data_vm_cpu_readings_vm_cpu_readings-file-*-of-195.csv.gz"))
    if CPU_FILES > 0:
        available = available[:CPU_FILES]
    if not available:
        raise FileNotFoundError(f"No Azure CPU shards found in {DATA.resolve()}")
    print(f"CPU shards selected: {len(available)} (CPU_FILES={CPU_FILES}; 0 means all local shards)")

    for i, p in enumerate(available, 1):
        
        print(f"\nProcessing file {i}/{len(available)}: {p.name}...")
        
        # Read CPU trace without header
        shard_rows = 0
        shard_parts = []
        for chunk in pd.read_csv(p, compression="gzip", header=None,
                                 usecols=[0, 1, 4], chunksize=250_000):
            shard_rows += len(chunk)
            chunk.columns = ["timestamp", "vm_id", "cpu_util_pct"]
            shard_parts.append(chunk)
        x = pd.concat(shard_parts, ignore_index=True)
        print(f"  Rows loaded: {shard_rows}")
        
        # Convert to numeric and handle errors
        x["timestamp"] = pd.to_numeric(x["timestamp"], errors="coerce")
        x["cpu_util_pct"] = pd.to_numeric(x["cpu_util_pct"], errors="coerce")
        x = x.dropna().drop_duplicates(["timestamp", "vm_id"])
        x = x[(x["timestamp"] >= 0) & np.isfinite(x["timestamp"])]
        x = x[(x["cpu_util_pct"] >= 0) & (x["cpu_util_pct"] <= 100)]
        
        # CRITICAL: CPU values are percentages (0-100), normalize to 0-1
        x["cpu_util"] = x["cpu_util_pct"] / 100.0
        x["cpu_util"] = x["cpu_util"].clip(0, 1)
        x = x[["timestamp", "vm_id", "cpu_util"]]
        
        print(f"  After cleaning: {len(x)} rows")
        print(f"  CPU range: {x['cpu_util'].min():.4f} - {x['cpu_util'].max():.4f}")
        
        total_rows_read += len(x)
        chunks.append(x)
    
    print(f"\nTotal rows across all files: {total_rows_read:,}")
    
    # Merge CPU trace with VM metadata
    cpu = pd.concat(chunks, ignore_index=True)
    print(f"Concatenated trace: {len(cpu)} rows")
    print(f"Unique VMs in trace: {cpu['vm_id'].nunique()}")
    
    cpu = cpu.merge(vm[["vm_id", "cores"]], on="vm_id", how="inner")
    print(f"After VM join: {len(cpu)} rows")
    
    # Aggregate into 5-minute time bins
    # Each interval is 5 minutes = 300 seconds
    cpu["step"] = (cpu["timestamp"] // (INTERVAL_MINUTES * 60)).astype(int)
    
    # Compute CPU demand: sum of (cores * utilization) for all VMs in interval
    cpu["cpu_demand"] = cpu["cores"] * cpu["cpu_util"]
    
    grouped = cpu.groupby("step").agg(
        cpu_demand=("cpu_demand", "sum"),
        vm_count=("vm_id", "nunique"),
        mean_cpu_util=("cpu_util", "mean"),
        p95_cpu_util=("cpu_util", lambda s: np.percentile(s, 95)),
        max_cpu_util=("cpu_util", "max"),
    ).reset_index()
    
    print(f"\nAggregated intervals: {len(grouped)}")
    
    # Filter: keep intervals with sufficient VM activity
    grouped = grouped[grouped.vm_count >= MIN_VMS_PER_STEP]
    grouped = grouped.sort_values("step").reset_index(drop=True)
    
    print(f"After filtering (min {MIN_VMS_PER_STEP} VMs/interval): {len(grouped)} intervals")
    
    # Limit to MAX_STEPS if needed
    if MAX_STEPS > 0:
        grouped = grouped.head(MAX_STEPS).reset_index(drop=True)
    print(f"After limiting to MAX_STEPS={MAX_STEPS or 'all'}: {len(grouped)} intervals")

    if len(grouped) < 20:
        print("WARNING: local Azure corpus has fewer than 20 intervals; results have limited statistical power.")
    
    # Normalize demand: scale to [0.02, 1.0] to make simulation challenging
    # but not impossibly hard. This is a normalized trace, not a physical size claim.
    demand = grouped["cpu_demand"].to_numpy()
    q99 = np.percentile(demand, 99)
    q50 = np.percentile(demand, 50)
    q95 = np.percentile(demand, 95)
    
    print(f"\nDemand statistics (raw CPU-cores):")
    print(f"  Min: {demand.min():.2f}")
    print(f"  Q50: {q50:.2f}")
    print(f"  Q95: {q95:.2f}")
    print(f"  Q99: {q99:.2f}")
    print(f"  Max: {demand.max():.2f}")
    
    grouped["normalized_demand"] = np.clip(demand / max(q99, 1e-9), 0.02, 1.0)
    grouped["previous_demand"] = grouped["normalized_demand"].shift(1).fillna(grouped["normalized_demand"].iloc[0])
    grouped["moving_average_demand"] = grouped["normalized_demand"].rolling(5, min_periods=1).mean()
    grouped["demand_volatility"] = grouped["normalized_demand"].rolling(5, min_periods=2).std().fillna(0.0)
    
    print(f"\nFinal trace:")
    print(f"  Intervals: {len(grouped)}")
    print(f"  Time span: {len(grouped) * INTERVAL_MINUTES / 60:.2f} hours")
    print(f"  Unique VMs observed: {cpu['vm_id'].nunique()}")
    print(f"  Total workload records: {len(cpu):,}")
    
    grouped.attrs["raw_records"] = int(len(cpu))
    grouped.attrs["unique_vms"] = int(cpu["vm_id"].nunique())
    grouped.attrs["timestamp_min"] = int(cpu["timestamp"].min())
    grouped.attrs["timestamp_max"] = int(cpu["timestamp"].max())
    grouped.attrs["cpu_shards"] = [p.name for p in available]
    return grouped
