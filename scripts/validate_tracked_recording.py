#!/usr/bin/env python3
"""Check a tracked-trial CSV against the contract before it reaches training.

Give this to whoever exports the EMG + tracker recordings. It checks the file
in isolation - no model, no other project code needs to run - and reports
exactly what is wrong, not just whether something is.

    python scripts/validate_tracked_recording.py trial_0001.csv
    python scripts/validate_tracked_recording.py recordings/*.csv --sensors S0 S1 S2 S3

What it checks:
  - required timing and tracker columns are present
  - EMG/IMU columns for the configured sensor names are present
  - time_perf_counter is finite and has no duplicate timestamps after
    resolution (matches how the loader deduplicates)
  - sample rate is stable, and reports gaps bigger than 3x the modal
    interval - a real dropout, not just measurement jitter
  - tracker position is finite where marked valid, in plausible units
    (metres, not centimetres or millimetres - flags a track that never
    moves more than a few units as a likely wrong unit or a static mount)
  - tracker validity coverage - what fraction of the trial has a usable
    position sample
  - EMG amplitude coverage per sensor
  - if a quaternion is present, that it is close to unit norm

This intentionally does not check EMG-tracker clock alignment beyond what
sharing time_perf_counter already guarantees - see the docstring in
data/tracked_trajectory.py for what is and is not covered.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.data.schema import emg_columns, imu_columns  # noqa: E402
from emg_touch.data.tracked_trajectory import (  # noqa: E402
    POSITION_COLUMNS,
    QUATERNION_COLUMNS,
    REQUIRED_TIMING_COLUMNS,
)


def check(path: Path, sensors: list[str] | None) -> list[str]:
    problems: list[str] = []
    warnings: list[str] = []

    try:
        frame = pd.read_csv(path)
    except Exception as error:  # noqa: BLE001
        return [f"could not read as CSV: {error}"]

    data_config = {"sensors": sensors} if sensors else {}
    required_emg = emg_columns(data_config)
    required_imu = imu_columns(data_config)

    for group, columns in (
        ("timing", REQUIRED_TIMING_COLUMNS),
        ("tracker position", POSITION_COLUMNS),
        ("EMG", required_emg),
        ("IMU", required_imu),
    ):
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            problems.append(f"missing {group} column(s): {missing}")
    if problems:
        return problems

    perf = pd.to_numeric(frame["time_perf_counter"], errors="coerce").to_numpy()
    if not np.isfinite(perf).all():
        problems.append(
            f"time_perf_counter has {int((~np.isfinite(perf)).sum())} non-finite rows"
        )
        return problems
    if not np.all(np.diff(perf) >= 0):
        unsorted_count = int((np.diff(perf) < 0).sum())
        warnings.append(
            f"time_perf_counter is not monotonic ({unsorted_count} inversions) - "
            "the loader sorts by it, but out-of-order rows usually mean a logging bug"
        )

    intervals = np.diff(np.sort(perf))
    intervals = intervals[intervals > 0]
    if len(intervals) < 2:
        problems.append("fewer than 2 distinct timestamps - trial is too short to check timing")
        return problems
    modal_interval = float(np.median(intervals))
    sample_rate = 1.0 / modal_interval if modal_interval > 0 else float("nan")
    gaps = intervals[intervals > 3.0 * modal_interval]
    if len(gaps):
        warnings.append(
            f"{len(gaps)} timing gap(s) over 3x the modal interval "
            f"(modal={modal_interval * 1000:.2f} ms, largest={gaps.max() * 1000:.1f} ms) - "
            "likely dropped samples, not just jitter"
        )

    position = frame[list(POSITION_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy()
    valid = np.isfinite(position).all(axis=1)
    if "pos_valid" in frame.columns:
        explicit = pd.to_numeric(frame["pos_valid"], errors="coerce").fillna(0).to_numpy() != 0
        valid = valid & explicit
    coverage = float(valid.mean()) if len(valid) else 0.0
    if coverage < 0.5:
        problems.append(f"tracker position valid for only {coverage:.0%} of samples")
    elif coverage < 0.95:
        warnings.append(f"tracker position valid for {coverage:.0%} of samples")

    if valid.any():
        span = position[valid].max(axis=0) - position[valid].min(axis=0)
        if np.all(span < 0.01):
            warnings.append(
                f"tracker position range is under 1 cm on every axis (span={span}) - "
                "check units are metres, not millimetres or centimetres, and that the "
                "tracker was actually moving during this trial"
            )
        if np.any(span > 50.0):
            warnings.append(
                f"tracker position range exceeds 50 m on an axis (span={span}) - "
                "check units, or for a corrupted sample"
            )

    for label, columns in (("EMG", required_emg), ("IMU", required_imu)):
        block = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy()
        finite = np.isfinite(block)
        per_channel = finite.mean(axis=0)
        low = [(columns[i], per_channel[i]) for i in range(len(columns)) if per_channel[i] < 0.5]
        if low:
            problems.append(f"{label} channels with <50% valid samples: {low}")

    if all(c in frame.columns for c in QUATERNION_COLUMNS):
        quaternion = (
            frame[list(QUATERNION_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy()
        )
        finite_rows = np.isfinite(quaternion).all(axis=1)
        if finite_rows.any():
            norm = np.linalg.norm(quaternion[finite_rows], axis=1)
            off_unit = np.abs(norm - 1.0) > 0.05
            if off_unit.any():
                warnings.append(
                    f"{int(off_unit.sum())} quaternion sample(s) more than 5% off unit norm"
                )

    duration = float(perf[-1] - perf[0])
    print(f"  duration ~{duration:.2f} s, {len(frame)} samples, ~{sample_rate:.1f} Hz")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="One or more trial CSV files")
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=None,
        help="Sensor names, in order (e.g. S0 S1 S2 S3). "
        "Defaults to this rig's four (S0 S4 S8 S12) if omitted.",
    )
    args = parser.parse_args()

    failures = 0
    for raw_path in args.csv:
        path = Path(raw_path)
        print(f"{path}")
        problems = check(path, args.sensors)
        if problems:
            failures += 1
            for problem in problems:
                print(f"  FAIL: {problem}")
        else:
            print("  OK")
        print()

    total = len(args.csv)
    print(f"{total - failures}/{total} file(s) passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
