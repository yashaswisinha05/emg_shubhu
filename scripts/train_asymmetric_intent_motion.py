#!/usr/bin/env python3
"""Train one asymmetrically routed causal EMG+IMU reach model.

The model simultaneously predicts touchscreen destination and complete 3D
reach. Cross-modal values remain available to both heads, but stop-gradient
boundaries prevent screen supervision from corrupting IMU motion and prevent
trajectory supervision from corrupting EMG intent.
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
from emg_touch.models.asymmetric_intent_motion import (  # noqa: E402
    AsymmetricIntentMotionModel,
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
            "--output-dir", "runs/asymmetric_intent_motion"
        ]
    base.make_distillation_window = complete.make_complete_reach_window
    base.milliseconds_to_samples = complete.milliseconds_to_samples
    base.evaluate = evaluate
    base.train_teacher = bridge.train_teacher
    channel.ChannelHorizonLatentDistillationModel = AsymmetricIntentMotionModel
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    print(
        "asymmetric intent-motion model: screen sees detached IMU factors; "
        "3D sees detached EMG intent; one model, one checkpoint, EMG+IMU only"
    )
    channel.main()


if __name__ == "__main__":
    main()
