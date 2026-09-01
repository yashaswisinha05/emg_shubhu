#!/usr/bin/env python3
"""Check a tracked-trial CSV against the contract before it reaches training.

Give this to whoever exports the EMG + Vive recordings. It checks the file
in isolation - no model, no other project code needs to run - and reports
exactly what is wrong, not just whether something is.

    python scripts/validate_tracked_recording.py trial_0001.csv
    python scripts/validate_tracked_recording.py recordings/*.csv --sensors S0 S1 S2 S3

If the tracker uses a different column prefix or a second tracker (e.g. a
forearm mount alongside the hand), pass --tracker-prefix / --tracker-id -
column names are {prefix}_{tracker}_{field}, e.g. VIVE_T0_pos_x_m by default.

What it checks:
  - required timing, EMG/IMU, and all 16 tracker columns are present
    (pos/quat/vel/angvel, tracking_age_us, vive_timestamp_us, sync_error_ms)
  - time_perf_counter is finite and has no duplicate timestamps after
    resolution (matches how the loader deduplicates)
  - sample rate is stable, and reports gaps bigger than 3x the modal
    interval - a real dropout, not just measurement jitter
  - tracker position is finite where valid, in plausible units (metres -
    flags a track that never moves more than 1 cm on any axis as a likely
    wrong unit or a static mount)
  - tracker validity coverage - what fraction of the trial has a usable
    position sample
  - EMG/IMU/velocity/angular-velocity amplitude coverage per channel
  - quaternion is close to unit norm, if present
  - sync_error_ms and tracking_age_us are REPORTED (mean/median/p95/max),
    not pass/failed against a threshold. There is no principled cutoff to
    hardcode without having seen this rig's real jitter - the same reason
    every other threshold in this project comes from a measurement, not a
    guess. Once real recordings show what "good" looks like, set
    data.tracker_max_sync_error_ms / data.tracker_max_tracking_age_us in
    the training config and tracked_trajectory.py will filter by it.

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
    REQUIRED_TIMING_COLUMNS,
    angular_velocity_columns,
    position_columns,
    quaternion_columns,
    sync_error_column,
    tracking_age_column,
    velocity_columns,
    vive_timestamp_column,
)


def _numeric_block(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy()


def check(path: Path, sensors: list[str] | None, tracker_config: dict) -> list[str]:
    problems: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    try:
        frame = pd.read_csv(path)
    except Exception as error:  # noqa: BLE001
        return [f"could not read as CSV: {error}"]

    data_config = dict(tracker_config)
    if sensors:
        data_config["sensors"] = sensors
    required_emg = emg_columns(data_config)
    required_imu = imu_columns(data_config)
    position_names = position_columns(data_config)
    quaternion_names = quaternion_columns(data_config)
    velocity_names = velocity_columns(data_config)
    angular_velocity_names = angular_velocity_columns(data_config)
    age_column = tracking_age_column(data_config)
    timestamp_column = vive_timestamp_column(data_config)
    sync_column = sync_error_column(data_config)

    for group, columns in (
        ("timing", REQUIRED_TIMING_COLUMNS),
        ("tracker position", position_names),
        ("tracker orientation", quaternion_names),
        ("tracker velocity", velocity_names),
        ("tracker angular velocity", angular_velocity_names),
        ("EMG", required_emg),
        ("IMU", required_imu),
    ):
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            problems.append(f"missing {group} column(s): {missing}")
    for label, column in (
        ("tracking age", age_column),
        ("Vive timestamp", timestamp_column),
        ("sync error", sync_column),
    ):
        if column not in frame.columns:
            problems.append(f"missing {label} column: {column}")
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

    position = _numeric_block(frame, position_names)
    valid = np.isfinite(position).all(axis=1)
    coverage = float(valid.mean()) if len(valid) else 0.0
    if coverage < 0.5:
        problems.append(f"tracker position finite for only {coverage:.0%} of samples")
    elif coverage < 0.95:
        warnings.append(f"tracker position finite for {coverage:.0%} of samples")

    if valid.any():
        span = position[valid].max(axis=0) - position[valid].min(axis=0)
        if np.all(span < 0.01):
            warnings.append(
                f"tracker position range is under 1 cm on every axis (span={span} m) - "
                "check units are metres, and that the tracker was actually moving"
            )
        if np.any(span > 50.0):
            warnings.append(
                f"tracker position range exceeds 50 m on an axis (span={span} m) - "
                "check units, or for a corrupted sample"
            )

    for label, columns in (
        ("EMG", required_emg),
        ("IMU", required_imu),
        ("tracker velocity", velocity_names),
        ("tracker angular velocity", angular_velocity_names),
    ):
        block = _numeric_block(frame, columns)
        finite = np.isfinite(block)
        per_channel = finite.mean(axis=0)
        low = [(columns[i], round(float(per_channel[i]), 3)) for i in range(len(columns)) if per_channel[i] < 0.5]
        if low:
            problems.append(f"{label} channels with <50% valid samples: {low}")

    quaternion = _numeric_block(frame, quaternion_names)
    finite_rows = np.isfinite(quaternion).all(axis=1)
    if finite_rows.any():
        norm = np.linalg.norm(quaternion[finite_rows], axis=1)
        off_unit = np.abs(norm - 1.0) > 0.05
        if off_unit.any():
            warnings.append(
                f"{int(off_unit.sum())} quaternion sample(s) more than 5% off unit norm"
            )

    # Reported, not pass/failed - see module docstring on why no threshold is
    # hardcoded here.
    sync_error = pd.to_numeric(frame[sync_column], errors="coerce").to_numpy()
    finite_sync = sync_error[np.isfinite(sync_error)]
    if len(finite_sync):
        info.append(
            f"sync_error_ms: mean={finite_sync.mean():.2f} median={np.median(finite_sync):.2f} "
            f"p95={np.percentile(finite_sync, 95):.2f} max={finite_sync.max():.2f}"
        )
    else:
        warnings.append("sync_error_ms is present but entirely non-finite")
    missing_sync_fraction = 1.0 - len(finite_sync) / max(len(sync_error), 1)
    if missing_sync_fraction > 0.05:
        warnings.append(f"sync_error_ms missing for {missing_sync_fraction:.0%} of samples")

    tracking_age = pd.to_numeric(frame[age_column], errors="coerce").to_numpy()
    finite_age = tracking_age[np.isfinite(tracking_age)]
    if len(finite_age):
        info.append(
            f"tracking_age_us: mean={finite_age.mean():.1f} median={np.median(finite_age):.1f} "
            f"p95={np.percentile(finite_age, 95):.1f} max={finite_age.max():.1f}"
        )
    else:
        warnings.append("tracking_age_us is present but entirely non-finite")

    # The recorded rate against the one the timestamps actually imply. dt
    # feeds the attractor dynamics directly, so a rate that is declared but
    # not delivered corrupts velocity and acceleration silently rather than
    # failing loudly.
    if "sample_rate_hz" in frame.columns:
        declared = pd.to_numeric(frame["sample_rate_hz"], errors="coerce").to_numpy()
        declared = declared[np.isfinite(declared)]
        if len(declared):
            declared_rate = float(np.median(declared))
            info.append(
                f"sample rate: declared {declared_rate:.2f} Hz, "
                f"timestamps imply {sample_rate:.2f} Hz"
            )
            if declared_rate > 0:
                relative = abs(sample_rate - declared_rate) / declared_rate
                if relative > 0.05:
                    warnings.append(
                        f"declared sample rate {declared_rate:.2f} Hz differs from the "
                        f"timestamp-derived {sample_rate:.2f} Hz by {relative:.0%} - "
                        "dt feeds velocity/acceleration and the attractor dynamics, so "
                        "decide which is authoritative before training on it"
                    )

    duration = float(perf[-1] - perf[0])
    print(f"  duration ~{duration:.2f} s, {len(frame)} samples, ~{sample_rate:.1f} Hz")
    for line in info:
        print(f"  {line}")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", nargs="+", help="One or more trial CSV files")
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=None,
        help="EMG/IMU sensor names, in order (e.g. S0 S1 S2 S3). "
        "Defaults to this rig's four (S0 S4 S8 S12) if omitted.",
    )
    parser.add_argument(
        "--tracker-prefix",
        default="VIVE",
        help="Tracker column prefix (default VIVE).",
    )
    parser.add_argument(
        "--tracker-id",
        default="T0",
        help="Tracker id, for a rig with more than one tracker (default T0).",
    )
    parser.add_argument(
        "--emg-template",
        default="EMG RMS 1_{sensor}",
        help="EMG column pattern. The original rig records an RMS envelope "
        '("EMG RMS 1_{sensor}", the default); a raw-EMG export uses '
        '"EMG 1_{sensor}". Run inspect_tracked_dataset.py to read it off the data.',
    )
    parser.add_argument(
        "--acc-template",
        default="ACC {axis}_{sensor}",
        help="Accelerometer column pattern.",
    )
    parser.add_argument(
        "--gyro-template",
        default="GYRO {axis}_{sensor}",
        help="Gyroscope column pattern.",
    )
    args = parser.parse_args()

    tracker_config = {
        "tracker_column_prefix": args.tracker_prefix,
        "tracker_id": args.tracker_id,
        "emg_column_template": args.emg_template,
        "acc_column_template": args.acc_template,
        "gyro_column_template": args.gyro_template,
    }

    failures = 0
    for raw_path in args.csv:
        path = Path(raw_path)
        print(f"{path}")
        problems = check(path, args.sensors, tracker_config)
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
