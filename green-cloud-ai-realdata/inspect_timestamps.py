import pandas as pd
import numpy as np

# Load CPU trace
cpu_file = r'data/azure_v2/trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-195.csv.gz'
df_cpu = pd.read_csv(cpu_file, compression='gzip', header=None, usecols=[0, 1, 4], nrows=1000000)
df_cpu.columns = ['timestamp', 'vm_id', 'cpu_pct']

print('=== TIMESTAMP ANALYSIS ===')
print(f'Min timestamp: {df_cpu["timestamp"].min()}')
print(f'Max timestamp: {df_cpu["timestamp"].max()}')
range_seconds = df_cpu["timestamp"].max() - df_cpu["timestamp"].min()
print(f'Range (seconds): {range_seconds}')
print(f'Range (hours): {range_seconds / 3600}')
print(f'Range (days): {range_seconds / 86400}')

# Check unique timestamps
df_cpu['step'] = (df_cpu['timestamp'] // 300).astype(int)
unique_steps = df_cpu['step'].nunique()
print(f'\nUnique 5-minute intervals: {unique_steps}')

print('\nFirst 20 unique timestamps:')
print(sorted(df_cpu['timestamp'].unique())[:20])
