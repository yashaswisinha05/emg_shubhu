#!/usr/bin/env python3
"""Train the teacher-guided soft-routed EMG+IMU reach model.

This is an isolated successor to the task-separated and asymmetric models.
It keeps the successful privileged teacher bridge, deterministic wearable
student, separate screen/3D heads, EMG-only auxiliary, and IMU path base.  It
replaces hard stop-gradient boundaries with configurable soft gradient scales
and adds modest signed-direction supervision for mirrored 3D reaches.

VIVE is used only by the training-only teacher and as a label.  Deployment is
``student_forward(emg, imu, time_mask)``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import train_channel_horizon_distillation_model as channel  # noqa: E402
from scripts import train_complete_reach_model as complete  # noqa: E402
from scripts import train_direction_aware_complete_reach as direction  # noqa: E402
from scripts import train_latent_distillation_model as base  # noqa: E402
from scripts import train_task_separated_complete_reach as task  # noqa: E402
from scripts import train_teacher_bridge_model as bridge  # noqa: E402
from emg_touch.models.soft_routed_complete_reach import (  # noqa: E402
    SoftRoutedCompleteReachModel,
)


def student_objective(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Joint Cartesian, screen, direction, and teacher-guidance objective."""
    combined = task.student_objective(
        outputs, teacher_outputs, window, config
    )
    settings = config["model"]["soft_routed_complete_reach"]
    fused = direction.direction_losses(outputs, window, config)
    emg = direction.direction_losses(outputs["emg_only"], window, config)
    combined["loss"] = (
        combined["loss"]
        + float(settings.get("endpoint_direction_weight", 0.15))
        * fused["direction_endpoint"]
        + float(settings.get("path_direction_weight", 0.10))
        * fused["direction_path"]
        + float(settings.get("axis_sign_weight", 0.10))
        * fused["direction_axis_sign"]
        + float(settings.get("velocity_direction_weight", 0.05))
        * fused["direction_velocity"]
        + float(settings.get("emg_direction_weight", 0.05))
        * (emg["direction_endpoint"] + 0.5 * emg["direction_axis_sign"])
    )
    combined.update({name: value.detach() for name, value in fused.items()})
    combined.update({
        f"emg_{name}": value.detach() for name, value in emg.items()
    })
    combined["screen_motion_gradient_scale"] = outputs[
        "factor_latent"
    ].new_tensor(settings.get("screen_motion_gradient_scale", 0.10)).detach()
    combined["trajectory_intent_gradient_scale"] = outputs[
        "trajectory_intent_gradient_scale"
    ].mean().detach()
    return combined


@torch.no_grad()
def evaluate(
    model: SoftRoutedCompleteReachModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    mean_target: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    metrics = task.evaluate(
        model,
        loader,
        config,
        context_samples,
        patch_length,
        evaluation_leads,
        canvas_tensor,
        mean_target,
        device,
    )
    metrics.update(direction.evaluate_direction(
        model,
        loader,
        config,
        context_samples,
        patch_length,
        evaluation_leads,
        canvas_tensor,
        device,
    ))
    return metrics


def _has_option(name: str) -> bool:
    return any(
        value == name or value.startswith(f"{name}=")
        for value in sys.argv[1:]
    )


def _option_value(name: str, default: str) -> str:
    arguments = sys.argv[1:]
    for index, value in enumerate(arguments):
        if value.startswith(f"{name}="):
            return value.split("=", 1)[1]
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return default


def main() -> None:
    if not _has_option("--config"):
        sys.argv[1:1] = [
            "--config", "configs/tracked_soft_routed_complete_reach.yaml"
        ]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = [
            "--output-dir", "runs/soft_routed_complete_reach"
        ]

    base.make_distillation_window = complete.make_complete_reach_window
    base.milliseconds_to_samples = complete.milliseconds_to_samples
    base.evaluate = evaluate
    base.train_teacher = bridge.train_teacher
    channel.ChannelHorizonLatentDistillationModel = SoftRoutedCompleteReachModel
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    print(
        "soft-routed reach: privileged teacher bridge + deterministic EMG+IMU "
        "student + separate screen/3D heads + attenuated cross-task gradients"
    )
    channel.main()

    output = Path(_option_value(
        "--output-dir", "runs/soft_routed_complete_reach"
    ))
    results_path = output / "results.json"
    if not results_path.exists():
        return
    with results_path.open("r", encoding="utf-8") as handle:
        test = json.load(handle).get("test", {})
    print("\n=== soft-routed complete-reach diagnostics ===")
    print(
        f"  screen late-window score: "
        f"{test.get('task_screen_score', float('nan')):.1f}px"
    )
    print(
        f"  complete 3D path: {test.get('task_fused_path_cm', float('nan')):.2f}cm | "
        f"IMU base: {test.get('task_imu_base_path_cm', float('nan')):.2f}cm | "
        f"gain: {test.get('task_path_gain_over_imu_cm', float('nan')):+.2f}cm"
    )
    print(
        f"  endpoint angle: "
        f"{test.get('direction_endpoint_angle_deg', float('nan')):.1f}° | "
        f"wrong-way: "
        f"{100.0 * test.get('direction_wrong_way', float('nan')):.1f}%"
    )
    for axis in direction.AXES:
        print(
            f"  {axis}: sign="
            f"{100.0 * test.get(f'direction_{axis}_sign_accuracy', float('nan')):5.1f}% "
            f"MAE={test.get(f'direction_{axis}_endpoint_mae_cm', float('nan')):5.2f}cm "
            f"r={test.get(f'direction_{axis}_correlation', float('nan')):+.3f}"
        )
    print("  checkpoints: best_screen.pt | best_3d.pt | best.pt (joint)")


if __name__ == "__main__":
    main()
