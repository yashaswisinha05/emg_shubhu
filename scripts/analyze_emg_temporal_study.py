#!/usr/bin/env python3
"""Measure the paired value added by each touch-relative EMG window."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


KEYS = ["configuration", "fold", "trial_id"]


def read_predictions(run_root: Path, filename: str) -> pd.DataFrame:
    frames = []
    pattern = f"*/fold-*/*/*/{filename}"
    for path in sorted(run_root.glob(pattern)):
        relative = path.relative_to(run_root)
        if len(relative.parts) != 5:
            continue
        configuration, fold_name, window_name, model_name, _ = relative.parts
        frame = pd.read_csv(path)
        frame["configuration"] = configuration
        frame["fold"] = int(fold_name.split("-", 1)[1])
        frame["requested_cutoff"] = window_name
        frame["model_kind"] = model_name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    duplicated = result.duplicated(
        ["configuration", "fold", "requested_cutoff", "model_kind", "trial_id"],
        keep=False,
    )
    if duplicated.any():
        examples = result.loc[
            duplicated,
            ["configuration", "fold", "requested_cutoff", "model_kind", "trial_id"],
        ].head(10)
        raise ValueError(f"Duplicate prediction records:\n{examples}")
    return result


def condition_frame(
    frame: pd.DataFrame,
    model_kind: str,
    prefix: str,
    metadata: bool = False,
    include_window: bool = False,
) -> pd.DataFrame:
    selected = frame.loc[frame["model_kind"] == model_kind].copy()
    columns = KEYS + [
        "pixel_error",
        "prediction_x",
        "prediction_y",
        "inside_target_box",
    ]
    if metadata:
        columns += [
            "subject",
            "requested_cutoff",
            "target_x",
            "target_y",
            "emg_window_samples",
        ]
    elif include_window:
        columns += ["requested_cutoff"]
    missing = set(columns) - set(selected.columns)
    if missing:
        raise ValueError(f"{model_kind} predictions lack columns {sorted(missing)}")
    selected = selected[columns]
    rename = {
        "pixel_error": f"{prefix}_error",
        "prediction_x": f"{prefix}_prediction_x",
        "prediction_y": f"{prefix}_prediction_y",
        "inside_target_box": f"{prefix}_inside_target_box",
    }
    if metadata or include_window:
        rename["requested_cutoff"] = "window"
    return selected.rename(columns=rename)


def paired_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    fusion = condition_frame(frame, "emg_residual", "fusion", metadata=True)
    # There is only one IMU baseline per fold, stored under causal_all.  Enforce
    # that invariant using the original frame before joining.
    causal_imu = frame.loc[
        (frame["model_kind"] == "imu_patch")
        & (frame["requested_cutoff"] == "causal_all")
    ]
    imu = condition_frame(causal_imu, "imu_patch", "imu")
    paired = fusion.merge(imu, on=KEYS, how="inner", validate="many_to_one")

    emg = condition_frame(
        frame, "emg_patch", "emg", include_window=True
    )
    if not emg.empty:
        # Match the EMG-only model to the same temporal window as fusion.
        paired = paired.merge(
            emg,
            on=KEYS + ["window"],
            how="left",
            validate="one_to_one",
        )
    else:
        paired["emg_error"] = np.nan
    if paired.empty:
        raise ValueError("No matched EMG-residual and causal-IMU predictions were found")

    paired["pixel_gain"] = paired["imu_error"] - paired["fusion_error"]
    paired["fusion_within_100"] = paired["fusion_error"] <= 100.0
    paired["imu_within_100"] = paired["imu_error"] <= 100.0
    paired["emg_within_100"] = paired["emg_error"] <= 100.0
    for prefix in ("fusion", "imu"):
        target_x_cell = np.clip((paired["target_x"] * 8).astype(int), 0, 7)
        target_y_cell = np.clip((paired["target_y"] * 5).astype(int), 0, 4)
        prediction_x_cell = np.clip(
            (paired[f"{prefix}_prediction_x"] * 8).astype(int), 0, 7
        )
        prediction_y_cell = np.clip(
            (paired[f"{prefix}_prediction_y"] * 5).astype(int), 0, 4
        )
        paired[f"{prefix}_screen_region_hit"] = (
            (target_x_cell == prediction_x_cell)
            & (target_y_cell == prediction_y_cell)
        )
    return paired


def bootstrap_subject_blocks(
    group: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    rng: np.random.Generator,
    repeats: int,
) -> tuple[float, float]:
    subjects = group["subject"].astype(str).unique()
    if len(subjects) < 2 or repeats <= 0:
        return float("nan"), float("nan")
    blocks = {subject: group.loc[group["subject"].astype(str) == subject] for subject in subjects}
    estimates = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        sample = pd.concat([blocks[subject] for subject in sampled_subjects], ignore_index=True)
        estimates[index] = statistic(sample)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def summarize_group(
    group: pd.DataFrame,
    rng: np.random.Generator,
    repeats: int,
) -> dict[str, float | int | str]:
    mean_ci = bootstrap_subject_blocks(
        group, lambda sample: float(sample["pixel_gain"].mean()), rng, repeats
    )
    median_ci = bootstrap_subject_blocks(
        group, lambda sample: float(sample["pixel_gain"].median()), rng, repeats
    )
    improved_ci = bootstrap_subject_blocks(
        group,
        lambda sample: float((sample["pixel_gain"] > 0.0).mean()),
        rng,
        repeats,
    )
    emg_errors = group["emg_error"].dropna()
    return {
        "held_out_trials": int(len(group)),
        "participants": int(group["subject"].nunique()),
        "emg_window_nonempty_fraction": float(
            (group["emg_window_samples"] > 0).mean()
        ),
        "median_emg_window_samples": float(group["emg_window_samples"].median()),
        "imu_median_pixel_error": float(group["imu_error"].median()),
        "fusion_median_pixel_error": float(group["fusion_error"].median()),
        "emg_median_pixel_error": (
            float(emg_errors.median()) if len(emg_errors) else float("nan")
        ),
        "difference_of_medians_px": float(
            group["imu_error"].median() - group["fusion_error"].median()
        ),
        "paired_mean_gain_px": float(group["pixel_gain"].mean()),
        "paired_mean_gain_ci95_low": mean_ci[0],
        "paired_mean_gain_ci95_high": mean_ci[1],
        "paired_median_gain_px": float(group["pixel_gain"].median()),
        "paired_median_gain_ci95_low": median_ci[0],
        "paired_median_gain_ci95_high": median_ci[1],
        "trajectory_fraction_improved": float((group["pixel_gain"] > 0.0).mean()),
        "trajectory_fraction_improved_ci95_low": improved_ci[0],
        "trajectory_fraction_improved_ci95_high": improved_ci[1],
        "imu_within_100_accuracy": float(group["imu_within_100"].mean()),
        "fusion_within_100_accuracy": float(group["fusion_within_100"].mean()),
        "within_100_gain_percentage_points": float(
            100.0
            * (group["fusion_within_100"].mean() - group["imu_within_100"].mean())
        ),
        "target_box_gain_percentage_points": float(
            100.0
            * (
                group["fusion_inside_target_box"].astype(float).mean()
                - group["imu_inside_target_box"].astype(float).mean()
            )
        ),
        "screen_region_gain_percentage_points": float(
            100.0
            * (
                group["fusion_screen_region_hit"].mean()
                - group["imu_screen_region_hit"].mean()
            )
        ),
    }


def summarize_all(
    paired: pd.DataFrame,
    split_name: str,
    rng: np.random.Generator,
    repeats: int,
) -> pd.DataFrame:
    rows = []
    for (configuration, window), group in paired.groupby(
        ["configuration", "window"], sort=True
    ):
        row: dict[str, float | int | str] = {
            "split": split_name,
            "configuration": configuration,
            "window": window,
        }
        row.update(summarize_group(group, rng, repeats))
        rows.append(row)
    return pd.DataFrame(rows)


def validation_selected_test(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    rng: np.random.Generator,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selections = []
    selected_test_parts = []
    for (configuration, fold), fold_validation in validation.groupby(
        ["configuration", "fold"], sort=True
    ):
        gains = (
            fold_validation.groupby("window", sort=True)["pixel_gain"]
            .mean()
            .sort_values(ascending=False)
        )
        window = str(gains.index[0])
        selections.append(
            {
                "configuration": configuration,
                "fold": int(fold),
                "selected_window": window,
                "validation_paired_mean_gain_px": float(gains.iloc[0]),
                "candidate_windows": int(len(gains)),
            }
        )
        selected = test.loc[
            (test["configuration"] == configuration)
            & (test["fold"] == fold)
            & (test["window"] == window)
        ].copy()
        if selected.empty:
            raise ValueError(
                f"No test predictions for validation-selected {configuration}/"
                f"fold-{fold}/{window}"
            )
        selected["selected_window"] = window
        selected_test_parts.append(selected)

    selection_frame = pd.DataFrame(selections)
    selected_test = pd.concat(selected_test_parts, ignore_index=True)
    rows = []
    for configuration, group in selected_test.groupby("configuration", sort=True):
        row: dict[str, float | int | str] = {
            "split": "test",
            "configuration": configuration,
            "window": "validation_selected_per_fold",
            "selected_windows_by_fold": json.dumps(
                {
                    str(int(fold)): str(window)
                    for fold, window in group[["fold", "selected_window"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                },
                sort_keys=True,
            ),
        }
        row.update(summarize_group(group, rng, repeats))
        rows.append(row)
    return selection_frame, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/emg_temporal_touch")
    parser.add_argument("--output-dir", default="evaluation/emg_temporal_touch")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--window",
        action="append",
        help="Analyze only this window; repeat as needed. Omit for every completed window.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_raw = read_predictions(run_root, "validation_predictions.csv")
    test_raw = read_predictions(run_root, "predictions.csv")
    if validation_raw.empty or test_raw.empty:
        raise ValueError(
            "Both validation_predictions.csv and predictions.csv are required. "
            "Run scripts/run_emg_temporal_study.py first."
        )
    validation = paired_predictions(validation_raw)
    test = paired_predictions(test_raw)
    if args.window:
        requested = set(args.window)
        available = set(validation["window"]) & set(test["window"])
        unknown = requested - available
        if unknown:
            raise ValueError(
                f"Requested windows have no complete predictions: {sorted(unknown)}; "
                f"available={sorted(available)}"
            )
        validation = validation.loc[validation["window"].isin(requested)].copy()
        test = test.loc[test["window"].isin(requested)].copy()
    rng = np.random.default_rng(args.seed)
    validation_results = summarize_all(
        validation, "validation", rng, args.bootstrap_repeats
    )
    test_results = summarize_all(test, "test", rng, args.bootstrap_repeats)
    results = pd.concat([validation_results, test_results], ignore_index=True)
    results = results.sort_values(
        ["split", "configuration", "paired_mean_gain_px"],
        ascending=[True, True, False],
    )
    results.to_csv(output_dir / "temporal_window_results.csv", index=False)

    selections, selected_test = validation_selected_test(
        validation, test, rng, args.bootstrap_repeats
    )
    selections.to_csv(output_dir / "fold_window_selection.csv", index=False)
    selected_test.to_csv(output_dir / "validation_selected_test_results.csv", index=False)

    print("TEST WINDOW RESULTS (positive gain means EMG helped)")
    display = test_results[
        [
            "configuration",
            "window",
            "paired_mean_gain_px",
            "paired_mean_gain_ci95_low",
            "paired_mean_gain_ci95_high",
            "fusion_median_pixel_error",
            "emg_median_pixel_error",
            "trajectory_fraction_improved",
        ]
    ].sort_values(["configuration", "paired_mean_gain_px"], ascending=[True, False])
    print(display.to_string(index=False))
    print(f"Wrote {output_dir / 'temporal_window_results.csv'}")
    print(f"Wrote {output_dir / 'fold_window_selection.csv'}")
    print(f"Wrote {output_dir / 'validation_selected_test_results.csv'}")


if __name__ == "__main__":
    main()
