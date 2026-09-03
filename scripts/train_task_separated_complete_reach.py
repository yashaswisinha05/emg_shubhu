#!/usr/bin/env python3
"""Train independently owned touchscreen and complete-3D heads.

The touchscreen head is trained from the EMG-owned intent representation.
The 3D head preserves the directly supervised IMU trajectory and learns only
a bounded intent-conditioned correction.  VIVE is used for teacher inputs and
labels, never by ``student_forward``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import train_channel_horizon_distillation_model as channel  # noqa: E402
from scripts import train_complete_reach_model as complete  # noqa: E402
from scripts import train_latent_distillation_model as base  # noqa: E402
from scripts import train_teacher_bridge_model as bridge  # noqa: E402
from emg_touch.models.task_separated_complete_reach import (  # noqa: E402
    TaskSeparatedCompleteReachModel,
)


_COMPLETE_OBJECTIVE = complete.student_objective
_COMPLETE_EVALUATE = complete.evaluate


def task_separation_losses(
    outputs: dict[str, Any], window: dict[str, Any], config: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Protect the IMU base and regularize EMG-conditioned corrections."""
    target = window["trajectory_target"]
    fused_error = torch.linalg.vector_norm(
        outputs["trajectory"] - target, dim=-1
    ).mean(dim=-1)
    base_error = torch.linalg.vector_norm(
        outputs["imu_base_trajectory"] - target, dim=-1
    ).mean(dim=-1)
    settings = config["model"]["task_separated_complete_reach"]
    margin = float(settings.get("base_preservation_margin_m", 0.0))
    preservation = F.relu(fused_error - base_error + margin).mean()
    correction = (
        outputs["path_correction"].square().mean()
        + outputs["endpoint_correction"].square().mean()
    )
    return {
        "task_base_preservation": preservation,
        "task_correction_regularization": correction,
    }


def student_objective(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    combined = _COMPLETE_OBJECTIVE(outputs, teacher_outputs, window, config)
    losses = task_separation_losses(outputs, window, config)
    settings = config["model"]["task_separated_complete_reach"]
    combined["loss"] = (
        combined["loss"]
        + float(settings.get("base_preservation_weight", 1.0))
        * losses["task_base_preservation"]
        + float(settings.get("correction_regularization_weight", 0.01))
        * losses["task_correction_regularization"]
    )
    combined.update({name: value.detach() for name, value in losses.items()})
    combined["task_path_correction_gate"] = (
        outputs["path_correction_gate"].mean().detach()
    )
    combined["task_endpoint_correction_gate"] = (
        outputs["endpoint_correction_gate"].mean().detach()
    )
    return combined


def _extend(store: dict[str, list[float]], name: str, values: torch.Tensor) -> None:
    store.setdefault(name, []).extend(values.detach().cpu().reshape(-1).tolist())


@torch.no_grad()
def evaluate_task_ownership(
    model: TaskSeparatedCompleteReachModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    steps = int(config["model"]["teacher_trajectory_steps"])
    limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    generator = np.random.default_rng(0)
    totals: dict[str, list[float]] = {}
    for batch in loader:
        if batch is None:
            continue
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        for lead in evaluation_leads:
            window = complete.make_complete_reach_window(
                batch, context_samples, patch_length, steps, generator,
                limit, velocity_scale, canvas_tensor, fixed_lead=lead,
            )
            if window is None:
                continue
            outputs = model.student_forward(
                window["emg"], window["imu"], window["time_mask"], sample=False
            )
            target = window["trajectory_target"]
            fused = 100.0 * torch.linalg.vector_norm(
                outputs["trajectory"] - target, dim=-1
            ).mean(dim=-1)
            imu_base = 100.0 * torch.linalg.vector_norm(
                outputs["imu_base_trajectory"] - target, dim=-1
            ).mean(dim=-1)
            _extend(totals, "task_fused_path_cm", fused)
            _extend(totals, "task_imu_base_path_cm", imu_base)
            _extend(totals, "task_path_gain_over_imu_cm", imu_base - fused)
            _extend(
                totals, "task_path_correction_gate",
                outputs["path_correction_gate"],
            )
            _extend(
                totals, "task_endpoint_correction_gate",
                outputs["endpoint_correction_gate"],
            )
    return {
        name: float(np.mean(values)) for name, values in totals.items() if values
    }


@torch.no_grad()
def evaluate(
    model: TaskSeparatedCompleteReachModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    mean_target: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    metrics = _COMPLETE_EVALUATE(
        model, loader, config, context_samples, patch_length, evaluation_leads,
        canvas_tensor, mean_target, device,
    )
    metrics.update(evaluate_task_ownership(
        model, loader, config, context_samples, patch_length, evaluation_leads,
        canvas_tensor, device,
    ))
    settings = config["model"]["task_separated_complete_reach"]
    rate = base.effective_rate(config)
    screen_values = []
    for milliseconds in settings.get("screen_selection_leads_ms", [0, 50, 100]):
        lead = complete.milliseconds_to_samples(float(milliseconds), rate)
        label = int(round(1000.0 * lead / rate))
        key = f"complete_screen_px_{label}ms"
        if key in metrics:
            screen_values.append(metrics[key])
    metrics["task_screen_score"] = float(np.mean(screen_values))
    metrics["task_3d_score"] = (
        metrics["complete_path_cm"]
        + float(settings.get("endpoint_score_weight", 0.5))
        * metrics["complete_endpoint_3d_cm"]
    )
    metrics["task_joint_score"] = (
        metrics["task_screen_score"]
        + float(settings.get("joint_3d_px_per_cm", 5.0))
        * metrics["task_3d_score"]
    )
    return metrics


def _has_option(name: str) -> bool:
    return any(value == name or value.startswith(f"{name}=") for value in sys.argv[1:])


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
            "--config", "configs/tracked_task_separated_complete_reach.yaml"
        ]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = [
            "--output-dir", "runs/task_separated_complete_reach"
        ]
    base.make_distillation_window = complete.make_complete_reach_window
    base.milliseconds_to_samples = complete.milliseconds_to_samples
    base.evaluate = evaluate
    base.train_teacher = bridge.train_teacher
    channel.ChannelHorizonLatentDistillationModel = TaskSeparatedCompleteReachModel
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    print(
        "task-separated reach: EMG-intent screen head | IMU-base 3D path + "
        "bounded detached-intent correction | VIVE labels only"
    )
    channel.main()

    output = Path(_option_value(
        "--output-dir", "runs/task_separated_complete_reach"
    ))
    results_path = output / "results.json"
    if not results_path.exists():
        return
    with results_path.open("r", encoding="utf-8") as handle:
        test = json.load(handle).get("test", {})
    print("\n=== task-separated test ===")
    print(
        f"  screen late-window score: {test.get('task_screen_score', float('nan')):.1f}px"
    )
    print(
        f"  corrected 3D path: {test.get('task_fused_path_cm', float('nan')):.2f}cm | "
        f"IMU base: {test.get('task_imu_base_path_cm', float('nan')):.2f}cm | "
        f"gain: {test.get('task_path_gain_over_imu_cm', float('nan')):+.2f}cm"
    )
    print(
        f"  correction gates: path={test.get('task_path_correction_gate', float('nan')):.3f} "
        f"endpoint={test.get('task_endpoint_correction_gate', float('nan')):.3f}"
    )
    print("  checkpoints: best_screen.pt | best_3d.pt | best.pt (joint)")


if __name__ == "__main__":
    main()
