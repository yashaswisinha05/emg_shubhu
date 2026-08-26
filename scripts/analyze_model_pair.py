#!/usr/bin/env python3
"""Paired, participant-blocked comparison of any two prediction models."""
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
    blocks = {
        subject: group.loc[group.subject == subject, column].to_numpy()
        for subject in subjects
    }
    estimates = np.empty(repeats)
    for index in range(repeats):
        selected = rng.choice(subjects, len(subjects), replace=True)
        sample = np.concatenate([blocks[subject] for subject in selected])
        estimates[index] = statistic(sample)
    return tuple(np.percentile(estimates, [2.5, 97.5]).astype(float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--base-kind", required=True)
    parser.add_argument("--candidate-kind", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    base = load_kind(run_root, args.base_kind)
    candidate = load_kind(run_root, args.candidate_kind)
    columns = KEYS + ["pixel_error", "inside_target_box", "cell_correct"]
    paired = candidate[columns].merge(
        base[columns],
        on=KEYS,
        suffixes=("_candidate", "_base"),
        validate="one_to_one",
    )
    paired["gain"] = paired.pixel_error_base - paired.pixel_error_candidate
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
                "base_model": args.base_kind,
                "candidate_model": args.candidate_kind,
                "base_median_pixel_error": group.pixel_error_base.median(),
                "candidate_median_pixel_error": group.pixel_error_candidate.median(),
                "paired_mean_gain_px": group.gain.mean(),
                "paired_mean_gain_ci95_low": mean_ci[0],
                "paired_mean_gain_ci95_high": mean_ci[1],
                "paired_median_gain_px": group.gain.median(),
                "paired_median_gain_ci95_low": median_ci[0],
                "paired_median_gain_ci95_high": median_ci[1],
                "trajectory_fraction_improved": (group.gain > 0).mean(),
                "within_100_gain_percentage_points": 100.0
                * (
                    (group.pixel_error_candidate <= 100).mean()
                    - (group.pixel_error_base <= 100).mean()
                ),
                "target_box_gain_percentage_points": 100.0
                * (
                    group.inside_target_box_candidate.astype(float).mean()
                    - group.inside_target_box_base.astype(float).mean()
                ),
                "grid_cell_gain_percentage_points": 100.0
                * (
                    group.cell_correct_candidate.astype(float).mean()
                    - group.cell_correct_base.astype(float).mean()
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        "paired_mean_gain_px", ascending=False
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
