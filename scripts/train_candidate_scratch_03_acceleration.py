#!/usr/bin/env python3
"""Stage 3/3: add acceleration dynamics to the candidate scratch model."""
from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import train_emg_acceleration_complete_reach as acceleration  # noqa: E402
from scripts import train_latent_distillation_model as base  # noqa: E402
from scripts import train_personalized_complete_reach as candidate  # noqa: E402


def _has(name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in sys.argv[1:])


def _value(name: str, default: str) -> str:
    for index, value in enumerate(sys.argv[1:]):
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
        if value == name and index + 1 < len(sys.argv[1:]):
            return sys.argv[1:][index + 1]
    return default


def main() -> None:
    if not _has("--initial-checkpoint"):
        sys.argv[1:1] = [
            "--initial-checkpoint", "runs/candidate_scratch_02_emg_residual/final.pt"
        ]
    if not _has("--config"):
        sys.argv[1:1] = [
            "--config", "configs/tracked_emg_acceleration_complete_reach.yaml"
        ]
    if not _has("--output-dir"):
        sys.argv[1:1] = ["--output-dir", "runs/candidate_scratch_03_acceleration"]
    base.build_experiment_loaders = candidate.build_candidate_loaders
    acceleration.main()
    candidate.save_candidate_calibration(
        _value("--output-dir", "runs/candidate_scratch_03_acceleration")
    )


if __name__ == "__main__":
    main()
