#!/usr/bin/env python3
"""Measure signal quality across a tracked dataset, not just its schema.

inspect_tracked_dataset.py answers "what columns exist"; this answers "is
what is in them usable". It samples several trials per session and reports
the things that quietly corrupt a dynamics model rather than failing loudly.

    python scripts/diagnose_tracked_quality.py "/path/to/emg_imu_vive" \\
        --emg-template "EMG 1_{sensor}"

What it measures, and why each one matters here:

  Tracker effective update rate. A Vive tracker runs at roughly 90-250 Hz
  while these rows are logged at ~1300 Hz, so the tracker columns are
  almost certainly forward-filled - each pose repeated across several rows.
  That is fine for position, but it makes the *reported* velocity a
  staircase, and differencing a staircase gives acceleration that is zero
  almost everywhere and enormous at the steps. The attractor readout
  r = x + (xddot + rho xdot)/eta consumes exactly that acceleration, so the
  repeat factor decides whether it has to be decimated to the true tracker
  rate first. Measured as the fraction of consecutive rows where position
  actually changes.

  tracking_age_us and sync_error_ms. Whether they carry information at all.
  A field that is always zero is a stub, not a clean signal, and quietly
  treating it as "no staleness" would be wrong. sync_error is separated
  into its constant part (a fixed offset, correctable) and its spread
  (jitter, not correctable) - only the second one is a hard limit on
  EMG-tracker alignment.

  Clock monotonicity. time_perf_counter comes from a monotonic clock, so
  inversions cannot be real time going backwards - they mean rows were
  written out of order. That matters beyond ordering: if EMG and tracker
  columns are filled by different threads, an out-of-order row may pair EMG
  from one instant with a pose from another, which no amount of sorting
  repairs.

  Sample delivery. Gaps against the modal interval, and whether the trial's
  row count matches its duration - bursty buffered delivery shows up as
  gaps alongside a total count that still looks right.

Reports per session and in aggregate; nothing is judged pass/fail, because
the thresholds that would matter here have to come from the capture
pipeline's own tolerances rather than from this script's assumptions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.data.schema import emg_columns  # noqa: E402
from emg_touch.data.tracked_trajectory import (  # noqa: E402
    position_columns,
    sync_error_column,
    tracking_age_column,
    velocity_columns,
)


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def measure_trial(path: Path, data_config: dict) -> dict | None:
    try:
        frame = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    if "time_perf_counter" not in frame.columns:
        return None

    result: dict = {"rows": len(frame)}
    perf = pd.to_numeric(frame["time_perf_counter"], errors="coerce").to_numpy()
    perf = finite(perf)
    if len(perf) < 3:
        return None

    deltas = np.diff(perf)
    result["inversions"] = int((deltas < 0).sum())
    result["duration_s"] = float(perf.max() - perf.min())
    positive = deltas[deltas > 0]
    if len(positive):
        modal = float(np.median(positive))
        result["row_rate_hz"] = 1.0 / modal if modal > 0 else np.nan
        result["gaps"] = int((positive > 3.0 * modal).sum())
        result["largest_gap_ms"] = float(positive.max() * 1000.0)

    # Effective tracker update rate: how often the pose actually changes.
    position_names = position_columns(data_config)
    if all(c in frame.columns for c in position_names):
        block = frame[list(position_names)].apply(pd.to_numeric, errors="coerce").to_numpy()
        usable = np.isfinite(block).all(axis=1)
        if usable.sum() > 3:
            rows = block[usable]
            changed = (np.abs(np.diff(rows, axis=0)) > 0).any(axis=1)
            fraction = float(changed.mean())
            result["pose_change_fraction"] = fraction
            if fraction > 0 and result.get("row_rate_hz"):
                result["tracker_rate_hz"] = result["row_rate_hz"] * fraction
                result["repeat_factor"] = 1.0 / fraction

    velocity_names = velocity_columns(data_config)
    if all(c in frame.columns for c in velocity_names):
        block = frame[list(velocity_names)].apply(pd.to_numeric, errors="coerce").to_numpy()
        usable = np.isfinite(block).all(axis=1)
        if usable.sum() > 3:
            rows = block[usable]
            changed = (np.abs(np.diff(rows, axis=0)) > 0).any(axis=1)
            result["velocity_change_fraction"] = float(changed.mean())

    age = tracking_age_column(data_config)
    if age in frame.columns:
        values = finite(pd.to_numeric(frame[age], errors="coerce").to_numpy())
        if len(values):
            result["age_max"] = float(values.max())
            result["age_nonzero_fraction"] = float((values != 0).mean())

    sync = sync_error_column(data_config)
    if sync in frame.columns:
        values = finite(pd.to_numeric(frame[sync], errors="coerce").to_numpy())
        if len(values):
            result["sync_mean_ms"] = float(values.mean())
            result["sync_std_ms"] = float(values.std())
            result["sync_max_ms"] = float(values.max())

    emg_names = emg_columns(data_config)
    present = [c for c in emg_names if c in frame.columns]
    if present:
        block = frame[present].apply(pd.to_numeric, errors="coerce").to_numpy()
        usable = block[np.isfinite(block).all(axis=1)]
        if len(usable):
            # Raw EMG is bipolar around zero; an RMS envelope is not. Which
            # one this is decides whether rectification is still needed.
            result["emg_mean"] = float(usable.mean())
            result["emg_min"] = float(usable.min())
            result["emg_negative_fraction"] = float((usable < 0).mean())
    return result


def aggregate(values: list[float]) -> str:
    array = np.array([v for v in values if v is not None and np.isfinite(v)])
    if not len(array):
        return "n/a"
    return f"{array.mean():.2f} (min {array.min():.2f}, max {array.max():.2f})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root")
    parser.add_argument("--trials-per-session", type=int, default=8)
    parser.add_argument("--sensors", nargs="+", default=None)
    parser.add_argument("--emg-template", default="EMG RMS 1_{sensor}")
    parser.add_argument("--tracker-prefix", default="VIVE")
    parser.add_argument("--tracker-id", default="T0")
    args = parser.parse_args()

    data_config: dict = {
        "emg_column_template": args.emg_template,
        "tracker_column_prefix": args.tracker_prefix,
        "tracker_id": args.tracker_id,
    }
    if args.sensors:
        data_config["sensors"] = args.sensors

    root = Path(args.root)
    sessions = sorted({p.parent for p in root.rglob("trial_*.csv")})
    if not sessions:
        print(f"no trial_*.csv under {root}", file=sys.stderr)
        sys.exit(2)

    everything: list[dict] = []
    print(f"sampling up to {args.trials_per_session} trial(s) from each of "
          f"{len(sessions)} session(s)\n")
    for session in sessions:
        trials = sorted(session.glob("trial_*.csv"))
        chosen = trials[:: max(1, len(trials) // args.trials_per_session)][
            : args.trials_per_session
        ]
        measured = [m for m in (measure_trial(p, data_config) for p in chosen) if m]
        if not measured:
            print(f"{session.name}: no readable trials")
            continue
        everything.extend(measured)

        def column(key: str) -> list[float]:
            return [m.get(key) for m in measured]

        rate = np.nanmean([m.get("row_rate_hz", np.nan) for m in measured])
        tracker = [m.get("tracker_rate_hz") for m in measured if m.get("tracker_rate_hz")]
        tracker_text = (
            f"{np.mean(tracker):7.1f} Hz" if tracker else "      n/a"
        )
        inversions = int(np.sum([m.get("inversions", 0) for m in measured]))
        rows = int(np.sum([m.get("rows", 0) for m in measured]))
        print(
            f"{session.name:16} rows/s {rate:7.1f} | tracker {tracker_text} | "
            f"inversions {inversions:4d}/{rows} | "
            f"sync {aggregate(column('sync_mean_ms'))} ms"
        )

    if not everything:
        return

    print()
    print("=== aggregate ===")

    def series(key: str) -> np.ndarray:
        return np.array(
            [m[key] for m in everything if key in m and np.isfinite(m[key])]
        )

    row_rate = series("row_rate_hz")
    tracker_rate = series("tracker_rate_hz")
    repeat = series("repeat_factor")
    if len(row_rate):
        print(f"  row rate            : {row_rate.mean():.1f} Hz "
              f"(min {row_rate.min():.1f}, max {row_rate.max():.1f})")
    if len(tracker_rate):
        print(f"  tracker update rate : {tracker_rate.mean():.1f} Hz "
              f"(min {tracker_rate.min():.1f}, max {tracker_rate.max():.1f})")
        print(f"  pose repeat factor  : {repeat.mean():.1f}x "
              f"- each pose is held across this many rows")
        if repeat.mean() > 1.5:
            print("    => tracker columns ARE forward-filled. Reported velocity is a")
            print("       staircase, so differencing it at the row rate gives spiky")
            print("       acceleration. Decimate to the tracker rate before the")
            print("       attractor readout, or smooth before differencing.")

    velocity_change = series("velocity_change_fraction")
    pose_change = series("pose_change_fraction")
    if len(velocity_change) and len(pose_change):
        print(f"  pose changes on     : {pose_change.mean() * 100:.1f}% of rows")
        print(f"  velocity changes on : {velocity_change.mean() * 100:.1f}% of rows")

    age_nonzero = series("age_nonzero_fraction")
    if len(age_nonzero):
        if age_nonzero.max() == 0.0:
            print("  tracking_age_us     : ALWAYS ZERO in every trial sampled - a stub,")
            print("                        not a clean signal. Do not filter on it.")
        else:
            print(f"  tracking_age_us     : nonzero on "
                  f"{age_nonzero.mean() * 100:.1f}% of samples")

    sync_mean = series("sync_mean_ms")
    sync_std = series("sync_std_ms")
    if len(sync_mean):
        print(f"  sync_error_ms       : offset {sync_mean.mean():.2f} ms, "
              f"jitter {sync_std.mean():.2f} ms, worst {series('sync_max_ms').max():.2f} ms")
        print("                        the offset is a fixed lag and correctable; the")
        print("                        jitter is the real limit on EMG-tracker alignment")

    inversions = series("inversions")
    rows = series("rows")
    if len(inversions) and inversions.sum() > 0:
        print(f"  clock inversions    : {inversions.sum():.0f} across "
              f"{rows.sum():.0f} rows ({inversions.sum() / rows.sum() * 100:.2f}%)")
        print("                        a monotonic clock cannot go backwards, so these are")
        print("                        rows written out of order. Sorting fixes the order,")
        print("                        but if EMG and tracker columns are filled by")
        print("                        different threads a row may pair EMG from one")
        print("                        instant with a pose from another - worth confirming")
        print("                        with whoever wrote the capture loop")

    negative = series("emg_negative_fraction")
    if len(negative):
        if negative.mean() > 0.1:
            print(f"  EMG                 : {negative.mean() * 100:.0f}% of samples "
                  "negative - raw bipolar signal, needs rectification and an envelope")
            print("                        before it is comparable to an RMS pipeline")
        else:
            print(f"  EMG                 : {negative.mean() * 100:.1f}% negative "
                  "- already an envelope/rectified")


if __name__ == "__main__":
    main()
