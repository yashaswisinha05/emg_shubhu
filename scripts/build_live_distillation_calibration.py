#!/usr/bin/env python3
"""Build EMG/IMU normalization for live latent-distillation inference.

The trained models use per-recording-session normalization. Live inference
therefore needs a calibration captured with the same wearer and sensor
placement; silently running without it would feed a different distribution
to the network. This command extracts exactly the same statistics as training
and writes only EMG/IMU normalization arrays to an NPZ file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    discover_trials,
    session_emg_scale,
    session_imu_statistics,
)


def matching_session_trials(root: str | Path, prefix: str) -> dict[str, list[Path]]:
    prefix = prefix.strip().lower()

    def matches(value: str) -> bool:
        lowered = value.lower()
        return (
            lowered == prefix
            or lowered.startswith(prefix + "_")
            or lowered.startswith(prefix + "-")
        )

    selected: dict[str, list[Path]] = {}
    for trials in discover_trials(root).values():
        for trial in trials:
            owner = next(
                (part for part in reversed(trial.parts) if matches(part)), None
            )
            if owner is not None:
                selected.setdefault(owner, []).append(trial)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--config", default="configs/tracked_channel_horizon_distillation.yaml"
    )
    parser.add_argument(
        "--session-prefix",
        required=True,
        help="One recording session, for example dev_a1",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    selected = matching_session_trials(args.root, args.session_prefix)
    if len(selected) != 1:
        raise SystemExit(
            f"{args.session_prefix!r} matched {len(selected)} sessions; "
            "use a prefix that identifies exactly one sensor placement"
        )
    session, trials = next(iter(selected.items()))
    emg_scale = session_emg_scale(trials, config["data"])
    imu_statistics = session_imu_statistics(trials, config["data"])
    if emg_scale is None or imu_statistics is None:
        raise SystemExit("could not compute calibration from the selected trials")
    imu_center, imu_scale = imu_statistics
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        emg_scale=emg_scale,
        imu_center=imu_center,
        imu_scale=imu_scale,
    )
    print(
        f"wrote {output} from {session} ({len(trials)} trials): "
        f"EMG {emg_scale.shape}, IMU {imu_center.shape}"
    )


if __name__ == "__main__":
    main()
