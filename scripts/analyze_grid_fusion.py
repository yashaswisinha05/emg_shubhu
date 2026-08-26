#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["configuration", "fold", "requested_cutoff", "trial_id", "subject"]


def load_kind(run_root: Path, kind: str) -> pd.DataFrame:
    frames = []
    for path in sorted(run_root.glob(f"*/fold-*/{kind}/predictions.csv")):
        frame = pd.read_csv(path)
        parts = path.relative_to(run_root).parts
        frame["configuration"] = parts[0]
        frame["fold"] = int(parts[1].split("-", 1)[1])
        frames.append(frame)
    if not frames:
        raise ValueError(f"No {kind} predictions found below {run_root}")
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(KEYS).any():
        raise ValueError(f"Duplicate {kind} predictions")
    return result


def subject_bootstrap(
    group: pd.DataFrame,
    column: str,
    statistic,
    rng: np.random.Generator,
    repeats: int,
) -> tuple[float, float]:
    subjects = group["subject"].unique()
    if len(subjects) < 2 or repeats <= 0:
        return float("nan"), float("nan")
    blocks = {subject: group.loc[group.subject == subject, column].to_numpy() for subject in subjects}
    estimates = np.empty(repeats)
    for index in range(repeats):
        selected = rng.choice(subjects, len(subjects), replace=True)
        sample = np.concatenate([blocks[subject] for subject in selected])
        estimates[index] = statistic(sample)
    return tuple(np.percentile(estimates, [2.5, 97.5]).astype(float))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired participant-blocked comparison of grid fusion versus grid IMU"
    )
    parser.add_argument("--run-root", default="runs/grid_point")
    parser.add_argument("--output", default="evaluation/grid_point/fusion_vs_imu.csv")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    imu = load_kind(run_root, "grid_imu")
    fusion = load_kind(run_root, "grid_fusion")
    imu_columns = KEYS + ["pixel_error", "inside_target_box", "cell_correct"]
    fusion_columns = KEYS + ["pixel_error", "inside_target_box", "cell_correct"]
    paired = fusion[fusion_columns].merge(
        imu[imu_columns],
        on=KEYS,
        suffixes=("_fusion", "_imu"),
        validate="one_to_one",
    )
    paired["gain"] = paired.pixel_error_imu - paired.pixel_error_fusion
    rng = np.random.default_rng(args.seed)
    rows = []
    for (configuration, cutoff), group in paired.groupby(
        ["configuration", "requested_cutoff"], sort=True
    ):
        mean_ci = subject_bootstrap(
            group, "gain", np.mean, rng, args.bootstrap_repeats
        )
        median_ci = subject_bootstrap(
            group, "gain", np.median, rng, args.bootstrap_repeats
        )
        rows.append(
            {
                "configuration": configuration,
                "requested_cutoff": cutoff,
                "held_out_trials": len(group),
                "participants": group.subject.nunique(),
                "imu_median_pixel_error": group.pixel_error_imu.median(),
                "fusion_median_pixel_error": group.pixel_error_fusion.median(),
                "paired_mean_gain_px": group.gain.mean(),
                "paired_mean_gain_ci95_low": mean_ci[0],
                "paired_mean_gain_ci95_high": mean_ci[1],
                "paired_median_gain_px": group.gain.median(),
                "paired_median_gain_ci95_low": median_ci[0],
                "paired_median_gain_ci95_high": median_ci[1],
                "trajectory_fraction_improved": (group.gain > 0).mean(),
                "within_100_gain_percentage_points": 100.0
                * (
                    (group.pixel_error_fusion <= 100).mean()
                    - (group.pixel_error_imu <= 100).mean()
                ),
                "target_box_gain_percentage_points": 100.0
                * (
                    group.inside_target_box_fusion.astype(float).mean()
                    - group.inside_target_box_imu.astype(float).mean()
                ),
                "grid_cell_gain_percentage_points": 100.0
                * (
                    group.cell_correct_fusion.astype(float).mean()
                    - group.cell_correct_imu.astype(float).mean()
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values("paired_mean_gain_px", ascending=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
