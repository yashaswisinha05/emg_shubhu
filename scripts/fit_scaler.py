#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from emg_touch.config import load_config
from emg_touch.data.full_trajectory import trajectory_analysis_interval
from emg_touch.data.manifest import load_manifest
from emg_touch.data.preprocessing import causal_median_filter, robust_statistics
from emg_touch.data.splits import subset_from_trial_ids
from emg_touch.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit robust channel scaling on the training partition only")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", help="Override split JSON")
    parser.add_argument("--output", help="Override scaler NPZ")
    parser.add_argument("--samples-per-trial", type=int, default=128)
    args = parser.parse_args()
    config = load_config(args.config)
    frame = load_manifest(config["paths"]["manifest"])
    split = load_json(args.split or config["paths"]["split_file"])
    train = subset_from_trial_ids(frame, split["train"])
    rng = np.random.default_rng(config["seed"])
    emg_values, imu_values = [], []
    kernel = int(config["data"].get("median_kernel", 1))
    use_log = bool(config["data"].get("emg_log1p", True))

    for row in tqdm(train.itertuples(index=False), total=len(train)):
        with np.load(row.cache_path) as cached:
            time_s = cached["time_s"]
            analysis_start, analysis_end = trajectory_analysis_interval(
                time_s,
                float(row.reaction_time_s),
                config["data"],
                float(getattr(row, "touch_time_s", float("nan"))),
            )
            duration = analysis_end - analysis_start
            minimum = float(config["data"].get("min_duration_s", 0.0))
            maximum = float(config["data"].get("max_duration_s", float("inf")))
            if duration < minimum or duration > maximum:
                continue
            eligible = np.flatnonzero(
                (time_s >= analysis_start) & (time_s <= analysis_end)
            )
            if len(eligible) == 0:
                continue
            count = min(args.samples_per_trial, len(eligible))
            indices = rng.choice(eligible, count, replace=False)
            emg = causal_median_filter(cached["emg"], kernel)[indices].astype(np.float64)
            imu = cached["imu"][indices].astype(np.float64)
            emg_mask = cached["emg_mask"][indices]
            imu_mask = cached["imu_mask"][indices]
        if use_log:
            emg = np.log1p(np.maximum(emg, 0.0))
        emg[~emg_mask] = np.nan
        imu[~imu_mask] = np.nan
        emg_values.append(emg)
        imu_values.append(imu)

    if not emg_values or not imu_values:
        raise ValueError("No valid training samples were available for scaler fitting")
    emg_center, emg_scale = robust_statistics(np.concatenate(emg_values, axis=0))
    imu_center, imu_scale = robust_statistics(np.concatenate(imu_values, axis=0))
    output = args.output or config["paths"]["scaler"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        emg_center=emg_center,
        emg_scale=emg_scale,
        imu_center=imu_center,
        imu_scale=imu_scale,
        emg_log1p=np.asarray(use_log),
    )
    print(f"Wrote training-only scaler to {output}")


if __name__ == "__main__":
    main()
