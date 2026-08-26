#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from emg_touch.config import load_config
from emg_touch.data.grid_trajectory import preprocess_grid_signals
from emg_touch.data.manifest import load_manifest
from emg_touch.data.preprocessing import robust_statistics
from emg_touch.data.splits import subset_from_trial_ids
from emg_touch.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit training-only scaling for EMG and configured IMU features"
    )
    parser.add_argument("--config", default="configs/grid_point.yaml")
    parser.add_argument("--split", help="Override split JSON")
    parser.add_argument("--output", help="Override scaler NPZ")
    parser.add_argument("--samples-per-trial", type=int, default=128)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = load_manifest(config["paths"]["manifest"])
    split = load_json(args.split or config["paths"]["split_file"])
    train = subset_from_trial_ids(manifest, split["train"])
    rng = np.random.default_rng(int(config["seed"]))
    minimum = float(config["data"].get("min_duration_s", 0.0))
    maximum = float(config["data"].get("max_duration_s", float("inf")))
    emg_values: list[np.ndarray] = []
    imu_values: list[np.ndarray] = []

    for row in tqdm(train.itertuples(index=False), total=len(train)):
        arrays = preprocess_grid_signals(row, config["data"], scaler=None)
        duration = float(arrays["duration_s"])
        if not np.isfinite(duration) or duration < minimum or duration > maximum:
            continue
        length = int(arrays["length"])
        count = min(args.samples_per_trial, length)
        indices = rng.choice(length, count, replace=False)
        emg = np.asarray(arrays["emg"])[indices].astype(np.float64)
        emg_mask = np.asarray(arrays["emg_mask"])[indices]
        imu = np.asarray(arrays["imu"])[indices].astype(np.float64)
        imu_mask = np.asarray(arrays["imu_mask"])[indices]
        emg[~emg_mask] = np.nan
        imu[~imu_mask] = np.nan
        emg_values.append(emg)
        imu_values.append(imu)

    if not emg_values or not imu_values:
        raise ValueError("No valid training samples were available for scaler fitting")
    emg_center, emg_scale = robust_statistics(np.concatenate(emg_values, axis=0))
    imu_center, imu_scale = robust_statistics(np.concatenate(imu_values, axis=0))
    output = Path(args.output or config["paths"]["scaler"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        emg_center=emg_center,
        emg_scale=emg_scale,
        imu_center=imu_center,
        imu_scale=imu_scale,
        emg_log1p=np.asarray(bool(config["data"].get("emg_log1p", True))),
    )
    print(
        f"Wrote grid scaler to {output.resolve()} "
        f"(EMG={len(emg_center)}, IMU={len(imu_center)})"
    )


if __name__ == "__main__":
    main()
