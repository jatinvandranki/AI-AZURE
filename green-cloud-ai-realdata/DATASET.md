# Dataset

This project uses Microsoft Azure Public Dataset V2 CPU trace files. CPU reading files are headerless; this implementation reads columns 0, 1, and 4 as timestamp, VM identifier, and average CPU utilization percentage.

CPU values are percentages in `[0, 100]` and are converted to utilization by dividing by 100. Invalid CPU values, malformed timestamps, missing identifiers, and duplicate `(timestamp, vm_id)` observations are reported.

Raw dataset files must be placed under `data/azure_v2` or passed with `--data-dir`. They are excluded from git.

## Official Source

The expected files are from the Microsoft Azure Public Dataset V2 GitHub release:

`https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2/`

The official Azure repository also lists these files in `AzurePublicDatasetLinksV2.txt`.

## Files Required For This Project's Full Mode

Current full-mode publication target: at least 2,000 genuine chronological workload intervals.

The loader automatically discovers all available CPU shards matching `trace_data_vm_cpu_readings_*.csv.gz`; it does not hardcode the number of shards. The current local data directory contains the VM table and multiple CPU shards. Run dataset check to produce the authoritative current counts.

Put every file in:

`data/azure_v2/`

## One-Command Download

From the project directory:

```bash
python setup_azure_data.py
```

Then check sufficiency:

```bash
python run_experiment.py --mode dataset-check
```

After the check says `SUFFICIENT`, run:

```bash
python run_experiment.py --mode full
```
