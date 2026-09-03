#!/usr/bin/env python3
"""Train direct deterministic EMG-screen and IMU-3D wearable heads.

The privileged teacher remains training-only for output distillation. Student
predictions bypass Gaussian sampling and KL matching. The proven deterministic
screen-coordinate adapter is retained; the direct IMU+EMG 3D route bypasses it.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import train_channel_horizon_distillation_model as channel  # noqa: E402
from scripts import train_complete_reach_model as complete  # noqa: E402
from scripts import train_latent_distillation_model as base  # noqa: E402
from scripts import train_task_separated_complete_reach as task  # noqa: E402
from scripts import train_teacher_bridge_model as bridge  # noqa: E402
from emg_touch.models.deterministic_complete_reach import (  # noqa: E402
    DeterministicCompleteReachModel,
)


student_objective = task.student_objective
evaluate = task.evaluate


def _has_option(name: str) -> bool:
    return any(value == name or value.startswith(f"{name}=") for value in sys.argv[1:])


def main() -> None:
    if not _has_option("--config"):
        sys.argv[1:1] = [
            "--config", "configs/tracked_task_separated_complete_reach.yaml"
        ]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = [
            "--output-dir", "runs/deterministic_complete_reach"
        ]
    base.make_distillation_window = complete.make_complete_reach_window
    base.milliseconds_to_samples = complete.milliseconds_to_samples
    base.evaluate = evaluate
    base.train_teacher = bridge.train_teacher
    channel.ChannelHorizonLatentDistillationModel = DeterministicCompleteReachModel
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    print(
        "deterministic student: raw EMG intent -> screen | raw IMU motion -> "
        "3D base + bounded EMG correction; no VAE sampling or KL in "
        "student prediction path"
    )
    channel.main()


if __name__ == "__main__":
    main()
