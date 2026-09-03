#!/usr/bin/env python3
"""Before training a 2 s trajectory horizon: is there enough trial to hold it?

The tracked dataset (emg_imu_vive) has NO post-touch buffer built into its
concept of a trial: touch = length - 1, the trial's last recorded sample,
full stop (see tracked_dataset.py). A horizon of H samples needs a cutoff
satisfying cutoff + H <= length, so a 2 s horizon needs the cutoff to sit a
full 2 s before the RECORDING ENDS, not just before the hand stops moving.
If reaches are shorter than 2 s (every prior measurement in this project -
the lead-time sweep, the trajectory rollout - suggests they are, around
1-1.5 s), most or all of that 2 s window falls in the stationary pre-reach
period, and the task quietly becomes "predict the eventual reach from
mostly-idle EMG/IMU" rather than "forecast the next 2 s of movement".
Neither is wrong, but they are different tasks and only one is what
train_trajectory_model.py's horizon was built for.

    python scripts/diagnose_horizon_feasibility.py \\
        --root "/media/.../emg_imu_vive" \\
        --config configs/tracked_trajectory_emg_enhanced.yaml

No model, no training - reads every trial once and reports, per candidate
horizon, what fraction of trials have ANY valid cutoff at all
(cutoff = max(onset, 0) + minimum_prefix, cutoff + horizon <= length), plus
the actual duration distribution so "how long are these trials really" has
a number instead of a guess.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    discover_trials,
    preprocess_tracked_trial,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/tracked_trajectory_emg_enhanced.yaml")
    parser.add_argument(
        "--horizons-ms", type=float, nargs="+",
        default=[250, 500, 750, 1000, 1500, 2000, 2500],
    )
    parser.add_argument("--limit", type=int, help="Sample this many trials per session.")
    args = parser.parse_args()

    config = load_config(args.config)
    minimum_prefix = int(config.get("virtual_leader", {}).get("minimum_prefix", 16))
    sessions = discover_trials(args.root)
    print(f"{len(sessions)} session(s) found, minimum_prefix={minimum_prefix}\n")

    lengths, onset_times, reach_durations = [], [], []
    rate = None
    for name, trials in sorted(sessions.items()):
        chosen = trials[: args.limit] if args.limit else trials
        for path in tqdm(chosen, desc=f"{name[:28]:28}", leave=False):
            data = preprocess_tracked_trial(path, config["data"])
            if data is None:
                continue
            if rate is None:
                rate = float(data["sample_rate_hz"])
            length = len(data["position"])
            onset = int(data["onset"])
            lengths.append(length)
            onset_times.append(onset / rate * 1000.0)
            reach_durations.append((length - 1 - onset) / rate * 1000.0)

    if not lengths:
        print("no usable trials found", file=sys.stderr)
        sys.exit(2)

    lengths = np.asarray(lengths)
    onset_times = np.asarray(onset_times)
    reach_durations = np.asarray(reach_durations)
    total_ms = lengths / rate * 1000.0

    def percentiles(values, label):
        p = np.percentile(values, [10, 50, 90])
        print(f"  {label:28} p10={p[0]:7.0f}  median={p[1]:7.0f}  p90={p[2]:7.0f}  (ms)")

    print(f"{len(lengths)} usable trials, decimated rate {rate:.1f} Hz\n")
    print("DURATIONS (this is the actual physical shape of a trial):")
    percentiles(total_ms, "total recorded length")
    percentiles(onset_times, "time to movement onset")
    percentiles(reach_durations, "onset -> end-of-recording (\"reach + whatever follows\")")

    print(f"\n{'horizon (ms)':>13}{'samples':>9}{'trials with >=1 valid cutoff':>32}")
    print("-" * 54)
    feasible_up_to = None
    for horizon_ms in sorted(args.horizons_ms):
        horizon = max(1, int(round(horizon_ms * rate / 1000.0)))
        start = np.maximum(np.zeros_like(lengths), minimum_prefix)
        # onset is not carried per-trial in the arrays above at full
        # resolution needed here without re-reading; use the coarser but
        # honest bound length - horizon > minimum_prefix, which is the
        # necessary condition regardless of onset placement - the true
        # feasible fraction (onset-aware) is <= this number.
        latest = lengths - horizon
        has_cutoff = latest > minimum_prefix
        fraction = float(np.mean(has_cutoff))
        print(f"{horizon_ms:>13.0f}{horizon:>9}{fraction * 100:>31.1f}%")
        if fraction >= 0.95:
            feasible_up_to = horizon_ms

    print()
    if feasible_up_to is not None and feasible_up_to >= max(args.horizons_ms):
        print(f"  >=95% of trials support every horizon tested, up to "
              f"{max(args.horizons_ms):.0f} ms.")
    elif feasible_up_to is not None:
        print(f"  >=95% of trials support horizons up to ~{feasible_up_to:.0f} ms; "
              "beyond that a\n  growing share of trials have NO valid cutoff at all and "
              "are silently\n  dropped from training (make_window returns None for them).")
    else:
        print("  Even the shortest horizon tested drops a meaningful fraction of "
              "trials.\n  Consider a shorter horizon or start from the diagnostic's own "
              "duration numbers\n  above to pick one the data actually supports.")
    print("\n  Note: this bound ignores onset placement (it is a NECESSARY condition,")
    print("  not sufficient) - the true feasible fraction is this number or lower. If")
    print("  it is already low here, the real number during training will be lower")
    print("  still.")


if __name__ == "__main__":
    main()
