#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.config import load_config
from src.dataset_requirements import inventory_data_dir, required_file_names


def parse_args():
    parser = argparse.ArgumentParser(description="Download real Microsoft Azure Public Dataset V2 files")
    parser.add_argument("--data-dir", default=None, help="Destination directory. Default: data/azure_v2")
    parser.add_argument("--target-intervals", type=int, default=None, help="Target genuine chronological intervals. Default: config publication target.")
    parser.add_argument("--cpu-files", default="auto", help="'auto' or an integer number of CPU shards from file 1 upward.")
    parser.add_argument("--dry-run", action="store_true", help="Print required files without downloading")
    return parser.parse_args()


def download(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"[exists] {target.name} ({target.stat().st_size / (1024 ** 3):.2f} GB)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"[download] {url}")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as fh:
            total = response.headers.get("Content-Length")
            total_int = int(total) if total else 0
            downloaded = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                fh.write(block)
                downloaded += len(block)
                if total_int:
                    print(f"\r  {downloaded / (1024 ** 3):.2f}/{total_int / (1024 ** 3):.2f} GB", end="")
        print()
        partial.replace(target)
        print(f"[saved] {target}")
    except urllib.error.URLError as exc:
        if partial.exists():
            partial.unlink()
        raise RuntimeError(f"Could not download {url}: {exc}") from exc


def main():
    args = parse_args()
    config = load_config(mode="full", data_dir=args.data_dir)
    if args.target_intervals:
        config.values["dataset"]["publication_target_intervals"] = args.target_intervals
        config.values["modes"]["full"]["min_intervals"] = args.target_intervals
    requirement = inventory_data_dir(config, current_intervals=0)
    if args.cpu_files != "auto":
        count = int(args.cpu_files)
        present = set(requirement.cpu_file_indices_present)
        requirement.required_cpu_files_total = count
        requirement.next_cpu_file_indices = [i for i in range(1, count + 1) if i not in present]
        requirement.additional_cpu_files_needed = len(requirement.next_cpu_file_indices)
    names = required_file_names(config, requirement)
    base = config.get("dataset")["release_base_url"]
    print("\nAZURE PUBLIC DATASET V2 DOWNLOAD PLAN\n-------------------------------------")
    print(f"Destination: {config.data_dir}")
    print(f"CPU shards target: {requirement.required_cpu_files_total}")
    print(f"Files to download now: {len(names)}")
    print(f"Approximate additional size: {requirement.approximate_required_download_gb} GB")
    for name in names:
        print(f"- {name}")
    if args.dry_run:
        return 0
    for name in names:
        download(base + name, config.data_dir / name)
    print("\nDownload complete. Run: python run_experiment.py --mode dataset-check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
