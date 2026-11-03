import pandas as pd

# Load raw sample
cpu_file = r'data/azure_v2/trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-195.csv.gz'

# Read first 50 rows
df = pd.read_csv(cpu_file, compression='gzip', header=None, nrows=50)
print("First 50 rows of CPU trace file:")
print(df.to_string(index=False))

print("\n=== SCHEMA HYPOTHESIS ===")
print("Columns appear to be:")
print("  0: timestamp (interval index)")
print("  1: vm_id")  
print("  2: min_cpu")
print("  3: max_cpu")
print("  4: avg_cpu")
print("\nTimestamp values (0, 1, 2, 3, 4...) suggest they are INTERVAL INDEXES")
print("Not seconds since epoch.")
print("\nIf each interval is 5 minutes:")
print("  20 intervals = 100 minutes = 1.67 hours")
print("  So this file spans only ~1.67 hours of data")
