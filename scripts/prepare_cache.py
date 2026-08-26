#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from emg_touch.config import load_config
from emg_touch.data.manifest import load_manifest
from emg_touch.data.preprocessing import csv_to_signal_arrays


def prepare_one(record: tuple[str, str, bool]) -> tuple[str, str]:
    csv_path, cache_path, overwrite = record
    destination = Path(cache_path)
    if destination.exists() and not overwrite:
        return "skipped", cache_path
    arrays = csv_to_signal_arrays(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return "written", cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache sorted, signal-only arrays from trial CSVs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    frame = load_manifest(config["paths"]["manifest"])
    records = [
        (row.csv_path, row.cache_path, args.overwrite)
        for row in frame.itertuples(index=False)
    ]
    counts = {"written": 0, "skipped": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for status, _ in tqdm(executor.map(prepare_one, records), total=len(records)):
            counts[status] += 1
    print(counts)


if __name__ == "__main__":
    main()

