#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_interval(
    values: np.ndarray,
    statistic,
    rng: np.random.Generator,
    repeats: int,
    blocks: np.ndarray | None = None,
) -> tuple[float, float]:
    if len(values) < 2 or repeats <= 0:
        return float("nan"), float("nan")
    estimates = np.empty(repeats, dtype=np.float64)
    if blocks is not None:
        unique_blocks = np.unique(blocks)
        if len(unique_blocks) < 2:
            return float("nan"), float("nan")
        grouped = {block: values[blocks == block] for block in unique_blocks}
    for index in range(repeats):
        if blocks is None:
            sample = rng.choice(values, size=len(values), replace=True)
        else:
            selected = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
            sample = np.concatenate([grouped[block] for block in selected])
        estimates[index] = statistic(sample)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def screen_region_accuracy(group: pd.DataFrame, grid_x: int, grid_y: int) -> float:
    target_x = np.clip((group["target_x"].to_numpy() * grid_x).astype(int), 0, grid_x - 1)
    target_y = np.clip((group["target_y"].to_numpy() * grid_y).astype(int), 0, grid_y - 1)
    prediction_x = np.clip(
        (group["prediction_x"].to_numpy() * grid_x).astype(int), 0, grid_x - 1
    )
    prediction_y = np.clip(
        (group["prediction_y"].to_numpy() * grid_y).astype(int), 0, grid_y - 1
    )
    return float(np.mean((target_x == prediction_x) & (target_y == prediction_y)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report held-out touch-prediction accuracy for each configuration"
    )
    parser.add_argument("predictions", nargs="+", help="One or more predictions.csv files")
    parser.add_argument("--output", default="configuration_comparison.csv")
    parser.add_argument("--grid", type=int, nargs=2, default=[8, 5], metavar=("X", "Y"))
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.concat([pd.read_csv(path) for path in args.predictions], ignore_index=True)
    required = {
        "trial_id",
        "configuration",
        "requested_cutoff",
        "target_x",
        "target_y",
        "prediction_x",
        "prediction_y",
        "pixel_error",
        "inside_target_box",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction files are missing columns: {sorted(missing)}")

    if "model_kind" not in frame:
        frame["model_kind"] = "unknown"
    identity = ["model_kind", "configuration", "requested_cutoff", "trial_id"]
    duplicates = frame.duplicated(identity, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, identity].head(10).to_dict("records")
        raise ValueError(
            "Duplicate held-out predictions were supplied. Each logical trial must appear once per "
            f"model/configuration/cutoff. Examples: {examples}"
        )

    rng = np.random.default_rng(args.seed)
    rows = []
    group_keys = ["model_kind", "configuration", "requested_cutoff"]
    for (model_kind, configuration, cutoff), group in frame.groupby(group_keys, sort=True):
        errors = group["pixel_error"].to_numpy(dtype=np.float64)
        hits = group["inside_target_box"].astype(float).to_numpy()
        blocks = (
            group["subject"].astype(str).to_numpy()
            if "subject" in group.columns
            else None
        )
        median_low, median_high = bootstrap_interval(
            errors, np.median, rng, args.bootstrap_repeats, blocks
        )
        hit_low, hit_high = bootstrap_interval(
            hits, np.mean, rng, args.bootstrap_repeats, blocks
        )
        normalized_error = np.sqrt(
            (group["prediction_x"].to_numpy() - group["target_x"].to_numpy()) ** 2
            + (group["prediction_y"].to_numpy() - group["target_y"].to_numpy()) ** 2
        )
        rows.append(
            {
                "model_kind": model_kind,
                "configuration": configuration,
                "requested_cutoff": cutoff,
                "held_out_trials": len(group),
                "participants": group["subject"].nunique() if "subject" in group else np.nan,
                "bootstrap_unit": "participant" if blocks is not None else "trajectory",
                "target_box_accuracy": float(np.mean(hits)),
                "target_box_accuracy_ci95_low": hit_low,
                "target_box_accuracy_ci95_high": hit_high,
                "screen_region_accuracy": screen_region_accuracy(
                    group, int(args.grid[0]), int(args.grid[1])
                ),
                "accuracy_within_50px": float(np.mean(errors <= 50.0)),
                "accuracy_within_100px": float(np.mean(errors <= 100.0)),
                "median_pixel_error": float(np.median(errors)),
                "median_pixel_error_ci95_low": median_low,
                "median_pixel_error_ci95_high": median_high,
                "mean_pixel_error": float(np.mean(errors)),
                "p90_pixel_error": float(np.percentile(errors, 90)),
                "mean_normalized_error": float(np.mean(normalized_error)),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["model_kind", "requested_cutoff", "target_box_accuracy", "median_pixel_error"],
        ascending=[True, True, False, True],
    )
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
