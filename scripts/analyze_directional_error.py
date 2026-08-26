#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def blocked_interval(
    frame: pd.DataFrame,
    column: str,
    rng: np.random.Generator,
    repeats: int,
) -> tuple[float, float]:
    subjects = frame["subject"].astype(str).unique()
    if len(subjects) < 2 or repeats <= 0:
        return float("nan"), float("nan")
    blocks = {
        subject: frame.loc[frame.subject.astype(str) == subject, column].to_numpy()
        for subject in subjects
    }
    estimates = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        selected = rng.choice(subjects, len(subjects), replace=True)
        sample = np.concatenate([blocks[subject] for subject in selected])
        estimates[index] = np.mean(sample)
    return tuple(np.percentile(estimates, [2.5, 97.5]).astype(float))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure absolute and inward-to-centre directional prediction bias"
    )
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--output", default="directional_error.csv")
    parser.add_argument("--canvas-width", type=float, default=1536.0)
    parser.add_argument("--canvas-height", type=float, default=774.0)
    parser.add_argument("--grid-width", type=int, default=8)
    parser.add_argument("--grid-height", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.concat(
        [pd.read_csv(path) for path in args.predictions], ignore_index=True
    )
    required = {
        "trial_id",
        "subject",
        "configuration",
        "model_kind",
        "target_x",
        "target_y",
        "prediction_x",
        "prediction_y",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction files are missing columns: {sorted(missing)}")
    if "requested_cutoff" not in frame:
        frame["requested_cutoff"] = "touch"
    identity = ["model_kind", "configuration", "requested_cutoff", "trial_id"]
    if frame.duplicated(identity).any():
        raise ValueError("Duplicate held-out predictions were supplied")

    width = float(args.canvas_width)
    height = float(args.canvas_height)
    frame["error_x_px"] = (frame.prediction_x - frame.target_x) * width
    frame["error_y_px"] = (frame.prediction_y - frame.target_y) * height
    target_from_center = np.column_stack(
        [(frame.target_x.to_numpy() - 0.5) * width,
         (frame.target_y.to_numpy() - 0.5) * height]
    )
    target_radius = np.linalg.norm(target_from_center, axis=1)
    inward_unit = -target_from_center / np.maximum(target_radius[:, None], 1e-8)
    errors = frame[["error_x_px", "error_y_px"]].to_numpy()
    frame["inward_error_px"] = np.sum(errors * inward_unit, axis=1)
    frame["is_inward"] = frame.inward_error_px > 0.0
    angles = np.arctan2(frame.error_y_px, frame.error_x_px)
    frame["angle_real"] = np.cos(angles)
    frame["angle_imag"] = np.sin(angles)

    rng = np.random.default_rng(args.seed)
    rows = []
    for (model_kind, configuration, cutoff), group in frame.groupby(
        ["model_kind", "configuration", "requested_cutoff"], sort=True
    ):
        x_ci = blocked_interval(
            group, "error_x_px", rng, args.bootstrap_repeats
        )
        y_ci = blocked_interval(
            group, "error_y_px", rng, args.bootstrap_repeats
        )
        inward_ci = blocked_interval(
            group, "inward_error_px", rng, args.bootstrap_repeats
        )
        resultant = np.hypot(group.angle_real.mean(), group.angle_imag.mean())
        row = {
            "model_kind": model_kind,
            "configuration": configuration,
            "requested_cutoff": cutoff,
            "held_out_trials": len(group),
            "participants": group.subject.nunique(),
            "mean_signed_x_error_px": group.error_x_px.mean(),
            "mean_signed_x_ci95_low": x_ci[0],
            "mean_signed_x_ci95_high": x_ci[1],
            "mean_signed_y_error_px": group.error_y_px.mean(),
            "mean_signed_y_ci95_low": y_ci[0],
            "mean_signed_y_ci95_high": y_ci[1],
            "horizontal_mae_px": group.error_x_px.abs().mean(),
            "vertical_mae_px": group.error_y_px.abs().mean(),
            "mean_inward_error_px": group.inward_error_px.mean(),
            "mean_inward_ci95_low": inward_ci[0],
            "mean_inward_ci95_high": inward_ci[1],
            "fraction_errors_inward": group.is_inward.mean(),
            "absolute_direction_concentration": resultant,
        }
        if {"target_cell", "predicted_cell"}.issubset(group.columns):
            grid_width = int(args.grid_width)
            grid_height = int(args.grid_height)
            target_x = group.target_cell.astype(int) % grid_width
            target_y = group.target_cell.astype(int) // grid_width
            predicted_x = group.predicted_cell.astype(int) % grid_width
            predicted_y = group.predicted_cell.astype(int) // grid_width
            target_edge = (
                target_x.isin([0, grid_width - 1])
                | target_y.isin([0, grid_height - 1])
            ).mean()
            predicted_edge = (
                predicted_x.isin([0, grid_width - 1])
                | predicted_y.isin([0, grid_height - 1])
            ).mean()
            row["target_edge_fraction"] = target_edge
            row["predicted_edge_fraction"] = predicted_edge
            row["edge_prediction_gap"] = predicted_edge - target_edge
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["model_kind", "configuration", "requested_cutoff"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
