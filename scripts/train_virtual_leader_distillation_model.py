#!/usr/bin/env python3
"""Train the factor-guided VAE with virtual-leader tracking losses.

This is a separate loss-only experiment built on the complete latent-
distillation trainer. The underlying model and deployment API are unchanged:
the student still receives only EMG, IMU, and a causal time mask. True VIVE
position is used only for training losses and offline evaluation.

Example:

    python scripts/train_virtual_leader_distillation_model.py \
      --root "/media/.../emg_imu_vive" \
      --config configs/tracked_latent_distillation_virtual_leader.yaml \
      --cache-dir artifacts/tracked_cache_posture \
      --session-prefixes dev_a1 dev_a2 dev_a3 dev_a4 \
      --device cuda --teacher-epochs 25 --epochs 50 --finetune-epochs 0 \
      --lead-window-ms 50 400 \
      --output-dir runs/latent_distillation_virtual_leader
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import train_latent_distillation_model as base  # noqa: E402
from emg_touch.virtual_leader_regularization import (  # noqa: E402
    virtual_leader_losses,
    weighted_virtual_leader_loss,
)


_BASE_TEACHER_OBJECTIVE = base.teacher_objective
_BASE_STUDENT_OBJECTIVE = base.student_objective


def _enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("virtual_leader_regularization", {}).get("enabled", True))


def teacher_objective(
    outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
    kl_weight: float,
) -> dict[str, torch.Tensor]:
    """Original teacher objective plus smooth, destination-driven motion."""
    combined = _BASE_TEACHER_OBJECTIVE(outputs, window, config, kl_weight)
    if not _enabled(config):
        return combined
    settings = config["virtual_leader_regularization"]
    additions = virtual_leader_losses(
        outputs["trajectory"],
        window["trajectory_target"],
        window["lead_samples"],
        base.effective_rate(config),
        settings,
    )
    combined["loss"] = combined["loss"] + weighted_virtual_leader_loss(
        additions, settings, "teacher"
    )
    combined.update({
        f"vl_{name}": value.detach() for name, value in additions.items()
    })
    return combined


def student_objective(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Original guided student objective plus joint tracking constraints."""
    combined = _BASE_STUDENT_OBJECTIVE(
        outputs, teacher_outputs, window, config
    )
    if not _enabled(config):
        return combined
    settings = config["virtual_leader_regularization"]
    rate = base.effective_rate(config)
    student_losses = virtual_leader_losses(
        outputs["trajectory"],
        window["trajectory_target"],
        window["lead_samples"],
        rate,
        settings,
    )
    imu_losses = virtual_leader_losses(
        outputs["imu_trajectory"],
        window["trajectory_target"],
        window["lead_samples"],
        rate,
        settings,
    )
    combined["loss"] = (
        combined["loss"]
        + weighted_virtual_leader_loss(student_losses, settings, "student")
        + weighted_virtual_leader_loss(imu_losses, settings, "imu")
    )
    combined.update({
        f"vl_{name}": value.detach() for name, value in student_losses.items()
    })
    combined.update({
        f"imu_vl_{name}": value.detach() for name, value in imu_losses.items()
    })
    return combined


def main() -> None:
    # Install the enhanced objectives only in this dedicated entry point.
    # Running train_latent_distillation_model.py continues to use its original
    # losses and reproduces the 228.6 px experiment.
    base.teacher_objective = teacher_objective
    base.student_objective = student_objective
    if not any(
        argument == "--config" or argument.startswith("--config=")
        for argument in sys.argv[1:]
    ):
        sys.argv[1:1] = [
            "--config",
            str(
                REPOSITORY_ROOT
                / "configs/tracked_latent_distillation_virtual_leader.yaml"
            ),
        ]
    print(
        "virtual-leader loss entry point: velocity + acceleration + endpoint "
        "+ destination-driven dynamics; deployment inputs remain EMG+IMU only"
    )
    base.__doc__ = __doc__
    base.main()


if __name__ == "__main__":
    main()
