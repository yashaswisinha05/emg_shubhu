#!/usr/bin/env python3
"""Is the tracker stream real at the row rate, or upsampled from a slower one?

diagnose_tracked_quality.py found the pose changing on 100% of rows at
~1259 Hz. A Vive tracker's lighthouse update is roughly 250 Hz, so it does
not natively produce a distinct pose per EMG sample. Either the capture
pipeline interpolates or extrapolates onto the EMG clock, or the values
carry float-level jitter that only looks like new data. Which one it is
decides whether the tracker columns can be differentiated at the row rate
at all.

    python scripts/diagnose_tracker_fidelity.py "/path/to/emg_imu_vive"

Three questions, each answered by a measurement rather than an assumption:

  1. Is the reported velocity independent of position, or just its
     derivative? If vel_x_mps is (to numerical precision) the finite
     difference of pos_x_m, then it carries no information position does
     not already have, and "use the measured velocity instead of
     differencing" - which is how tracked_virtual_leader.py currently
     justifies its velocity input - buys nothing. Compared against both a
     central and a backward difference, since a pipeline could use either.

  2. Is position piecewise-linear? Linear interpolation from a slower
     source leaves the second difference at essentially zero everywhere
     except at the knots. Genuine motion sampled at 1259 Hz does not do
     that. The fraction of near-zero second differences, plus the spacing
     between the non-zero ones, recovers the original rate if there is one.

  3. How bad is the sync-error tail? The mean is ~3 ms but the worst seen
     was 89 ms - longer than the 40-80 ms electromechanical delay that
     makes EMG predictive in the first place. A mean says nothing about how
     many samples are unusable, so this reports the tail directly.

Nothing here is pass/fail. The point is to know what the signal is before
building on it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.data.tracked_trajectory import (  # noqa: E402
    position_columns,
    sync_error_column,
    velocity_columns,
)


def relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    """Median |a-b| / scale, robust to the odd outlier sample."""
    scale = np.median(np.abs(right)) + 1e-12
    return float(np.median(np.abs(left - right)) / scale)


def analyse(path: Path, data_config: dict) -> dict | None:
    try:
        frame = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    position_names = position_columns(data_config)
    velocity_names = velocity_columns(data_config)
    if not all(c in frame.columns for c in position_names):
        return None
    if "time_perf_counter" not in frame.columns:
        return None

    perf = pd.to_numeric(frame["time_perf_counter"], errors="coerce").to_numpy()
    position = frame[list(position_names)].apply(pd.to_numeric, errors="coerce").to_numpy()
    usable = np.isfinite(perf) & np.isfinite(position).all(axis=1)
    perf, position = perf[usable], position[usable]
    if len(perf) < 50:
        return None
    order = np.argsort(perf, kind="stable")
    perf, position = perf[order], position[order]

    intervals = np.diff(perf)
    positive = intervals[intervals > 0]
    dt = float(np.median(positive)) if len(positive) else np.nan
    result: dict = {"rate_hz": 1.0 / dt if dt else np.nan, "rows": len(perf)}

    # 1. reported velocity against derivatives of position
    if all(c in frame.columns for c in velocity_names):
        velocity = (
            frame[list(velocity_names)].apply(pd.to_numeric, errors="coerce").to_numpy()
        )
        velocity = velocity[usable][order]
        if np.isfinite(velocity).all():
            central = np.gradient(position, axis=0) / dt
            backward = np.zeros_like(position)
            backward[1:] = np.diff(position, axis=0) / dt
            interior = slice(2, -2)
            result["vel_vs_central"] = relative_difference(
                velocity[interior], central[interior]
            )
            result["vel_vs_backward"] = relative_difference(
                velocity[interior], backward[interior]
            )
            result["vel_scale"] = float(np.median(np.abs(velocity)))

    # 2. piecewise-linearity of position
    second = np.diff(position, n=2, axis=0)
    magnitude = np.abs(second).max(axis=1)
    if len(magnitude) > 10 and magnitude.max() > 0:
        # "Flat" relative to the trial's own curvature scale, so this does
        # not depend on the units or how fast the person moved.
        threshold = 0.01 * np.percentile(magnitude, 99)
        flat = magnitude <= threshold
        result["flat_second_difference_fraction"] = float(flat.mean())
        knots = np.flatnonzero(~flat)
        if len(knots) > 5:
            spacing = np.diff(knots)
            result["knot_spacing_median"] = float(np.median(spacing))
            result["knot_spacing_mode"] = float(
                np.bincount(spacing[spacing < 64]).argmax()
            ) if (spacing < 64).any() else np.nan

    # 3. sync-error tail
    sync = sync_error_column(data_config)
    if sync in frame.columns:
        values = pd.to_numeric(frame[sync], errors="coerce").to_numpy()
        values = values[np.isfinite(values)]
        if len(values):
            result["sync_median"] = float(np.median(values))
            for limit in (10.0, 20.0, 40.0):
                result[f"sync_over_{limit:.0f}ms"] = float((values > limit).mean())
            result["sync_max"] = float(values.max())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root")
    parser.add_argument("--trials-per-session", type=int, default=4)
    parser.add_argument("--tracker-prefix", default="VIVE")
    parser.add_argument("--tracker-id", default="T0")
    args = parser.parse_args()

    data_config = {
        "tracker_column_prefix": args.tracker_prefix,
        "tracker_id": args.tracker_id,
    }
    root = Path(args.root)
    sessions = sorted({p.parent for p in root.rglob("trial_*.csv")})
    if not sessions:
        print(f"no trial_*.csv under {root}", file=sys.stderr)
        sys.exit(2)

    records: list[dict] = []
    for session in sessions:
        trials = sorted(session.glob("trial_*.csv"))
        chosen = trials[:: max(1, len(trials) // args.trials_per_session)][
            : args.trials_per_session
        ]
        for path in chosen:
            record = analyse(path, data_config)
            if record:
                records.append(record)
    if not records:
        print("nothing analysable", file=sys.stderr)
        sys.exit(2)

    def series(key: str) -> np.ndarray:
        return np.array([r[key] for r in records if key in r and np.isfinite(r[key])])

    print(f"analysed {len(records)} trial(s) across {len(sessions)} session(s)")
    print(f"row rate: {series('rate_hz').mean():.1f} Hz")
    print()

    print("=== 1. is the reported velocity independent of position? ===")
    central = series("vel_vs_central")
    backward = series("vel_vs_backward")
    if not len(central):
        print("  no velocity columns found")
    else:
        print(f"  vs central difference of position : {central.mean():.2e} relative")
        print(f"  vs backward difference of position: {backward.mean():.2e} relative")
        best = min(central.mean(), backward.mean())
        if best < 1e-6:
            print("  => velocity IS the derivative of position, to numerical precision.")
            print("     It carries no information position does not already have, so")
            print("     feeding it to the model instead of differencing buys nothing -")
            print("     the claim that it avoids a noisy differentiation does not hold.")
        elif best < 1e-2:
            print("  => velocity closely tracks the derivative but is not identical:")
            print("     probably the same source with filtering or a different stencil.")
            print("     Mildly useful; not an independent measurement.")
        else:
            print("  => velocity differs substantially from the derivative, so it is a")
            print("     genuinely separate estimate (tracker IMU fusion) and is worth")
            print("     using directly rather than differencing position.")

    print()
    print("=== 2. is position piecewise-linear (upsampled from a slower rate)? ===")
    flat = series("flat_second_difference_fraction")
    if not len(flat):
        print("  not measurable")
    else:
        print(f"  second difference is ~flat on {flat.mean() * 100:.1f}% of samples")
        spacing = series("knot_spacing_median")
        if len(spacing):
            print(f"  median spacing between curvature knots: {spacing.mean():.2f} samples")
        if flat.mean() > 0.5:
            rate = series("rate_hz").mean()
            factor = 1.0 / max(1.0 - flat.mean(), 1e-6)
            print(f"  => position is largely piecewise-linear. It is being interpolated")
            print(f"     onto the EMG clock from roughly {rate / factor:.0f} Hz.")
            print("     Acceleration at the row rate is then an artifact - near zero")
            print("     inside segments and spiking at the knots - so the attractor")
            print("     readout must run at the true tracker rate, not the row rate.")
        else:
            print("  => position is NOT piecewise-linear: curvature is present")
            print("     throughout, consistent with a genuine per-sample pose stream")
            print("     (SteamVR pose prediction, or IMU-fused extrapolation).")

    print()
    print("=== 3. sync-error tail ===")
    median = series("sync_median")
    if len(median):
        print(f"  median {median.mean():.2f} ms, worst {series('sync_max').max():.2f} ms")
        for limit in (10, 20, 40):
            over = series(f"sync_over_{limit}ms")
            if len(over):
                print(f"  samples over {limit:>2d} ms: {over.mean() * 100:.3f}%")
        print("  the electromechanical delay that makes EMG predictive is 40-80 ms, so")
        print("  samples whose misalignment approaches that are the ones to drop -")
        print("  set data.tracker_max_sync_error_ms from the tail above")


if __name__ == "__main__":
    main()
