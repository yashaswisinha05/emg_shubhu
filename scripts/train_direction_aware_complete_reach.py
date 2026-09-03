#!/usr/bin/env python3
"""Train complete-reach prediction with explicit 3D direction supervision.

This isolated successor keeps the stable onset-to-touch path target and true
0 ms observation from ``train_complete_reach_model.py``.  It additionally:

* sends a gated EMG-owned screen-intent residual into the 3D decoder;
* classifies negative/stationary/positive displacement on x, y, and z;
* penalizes endpoint, net-path, and stepwise velocity direction errors;
* reports per-axis sign accuracy, MAE, and correlation.

The deployment API remains ``student_forward(emg, imu, time_mask)``.  VIVE
direction and trajectory are labels only.
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
from emg_touch.models.direction_aware_complete_reach import (  # noqa: E402
    DirectionAwareCompleteReachModel,
)


_COMPLETE_STUDENT_OBJECTIVE = complete.student_objective
_COMPLETE_EVALUATE = complete.evaluate
AXES = ("x", "y", "z")


def axis_direction_labels(
    displacement: torch.Tensor, stationary_threshold_m: float
) -> torch.Tensor:
    """Map signed XYZ displacement to negative/stationary/positive classes."""
    labels = torch.ones_like(displacement, dtype=torch.long)
    labels = torch.where(displacement < -stationary_threshold_m, 0, labels)
    labels = torch.where(displacement > stationary_threshold_m, 2, labels)
    return labels


def _cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor | None = None,
) -> torch.Tensor:
    cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-6)
    loss = 1.0 - cosine
    if validity is None:
        return loss.mean()
    weights = validity.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def direction_losses(
    outputs: dict[str, Any],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    settings = config["model"]["direction_aware_complete_reach"]
    endpoint_target = window["endpoint_3d_target"]
    stationary = float(settings.get("stationary_threshold_m", 0.015))
    labels = axis_direction_labels(endpoint_target, stationary)
    axis_sign = F.cross_entropy(
        outputs["axis_direction_logits"].reshape(-1, 3), labels.reshape(-1)
    )
    endpoint_direction = _cosine_loss(outputs["endpoint_3d"], endpoint_target)

    predicted_path = outputs["trajectory"]
    target_path = window["trajectory_target"]
    path_direction = _cosine_loss(
        predicted_path[:, -1] - predicted_path[:, 0],
        target_path[:, -1] - target_path[:, 0],
    )
    predicted_velocity = torch.diff(predicted_path, dim=1)
    target_velocity = torch.diff(target_path, dim=1)
    velocity_threshold = float(settings.get("velocity_threshold_m", 0.002))
    valid_velocity = target_velocity.norm(dim=-1) > velocity_threshold
    velocity_direction = _cosine_loss(
        predicted_velocity, target_velocity, valid_velocity
    )
    return {
        "direction_endpoint": endpoint_direction,
        "direction_path": path_direction,
        "direction_axis_sign": axis_sign,
        "direction_velocity": velocity_direction,
    }


def student_objective(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    combined = _COMPLETE_STUDENT_OBJECTIVE(
        outputs, teacher_outputs, window, config
    )
    settings = config["model"]["direction_aware_complete_reach"]
    fused = direction_losses(outputs, window, config)
    emg = direction_losses(outputs["emg_only"], window, config)
    combined["loss"] = (
        combined["loss"]
        + float(settings.get("endpoint_direction_weight", 0.50))
        * fused["direction_endpoint"]
        + float(settings.get("path_direction_weight", 0.25))
        * fused["direction_path"]
        + float(settings.get("axis_sign_weight", 0.25))
        * fused["direction_axis_sign"]
        + float(settings.get("velocity_direction_weight", 0.20))
        * fused["direction_velocity"]
        + float(settings.get("emg_direction_weight", 0.20))
        * (
            emg["direction_endpoint"]
            + 0.5 * emg["direction_axis_sign"]
        )
    )
    combined.update({name: value.detach() for name, value in fused.items()})
    combined.update({f"emg_{name}": value.detach() for name, value in emg.items()})
    combined["intent_to_motion_gate"] = (
        outputs["intent_to_motion_gate"].mean().detach()
    )
    return combined


def _extend(
    totals: dict[str, list[float]], name: str, values: torch.Tensor
) -> None:
    totals.setdefault(name, []).extend(
        values.detach().cpu().reshape(-1).tolist()
    )


def _angle_degrees(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(first, second, dim=-1, eps=1e-6)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


@torch.no_grad()
def evaluate_direction(
    model: DirectionAwareCompleteReachModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rate = base.effective_rate(config)
    steps = int(config["model"]["teacher_trajectory_steps"])
    limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    settings = config["model"]["direction_aware_complete_reach"]
    stationary = float(settings.get("stationary_threshold_m", 0.015))
    velocity_threshold = float(settings.get("velocity_threshold_m", 0.002))
    generator = np.random.default_rng(0)
    totals: dict[str, list[float]] = {}
    predicted_axes: list[list[float]] = [[], [], []]
    true_axes: list[list[float]] = [[], [], []]

    for batch in loader:
        if batch is None:
            continue
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        for lead in evaluation_leads:
            window = complete.make_complete_reach_window(
                batch,
                context_samples,
                patch_length,
                steps,
                generator,
                limit,
                velocity_scale,
                canvas_tensor,
                fixed_lead=lead,
            )
            if window is None:
                continue
            outputs = model.student_forward(
                window["emg"], window["imu"], window["time_mask"], sample=False
            )
            predicted = outputs["endpoint_3d"]
            target = window["endpoint_3d_target"]
            labels = axis_direction_labels(target, stationary)
            classes = outputs["axis_direction_logits"].argmax(dim=-1)
            endpoint_angle = _angle_degrees(predicted, target)
            path_angle = _angle_degrees(
                outputs["trajectory"][:, -1] - outputs["trajectory"][:, 0],
                window["trajectory_target"][:, -1]
                - window["trajectory_target"][:, 0],
            )
            predicted_velocity = torch.diff(outputs["trajectory"], dim=1)
            target_velocity = torch.diff(window["trajectory_target"], dim=1)
            valid_velocity = target_velocity.norm(dim=-1) > velocity_threshold
            velocity_angle = _angle_degrees(
                predicted_velocity, target_velocity
            )
            label = int(round(1000.0 * lead / rate))
            _extend(totals, "direction_endpoint_angle_deg", endpoint_angle)
            _extend(
                totals, f"direction_endpoint_angle_{label}ms_deg", endpoint_angle
            )
            _extend(totals, "direction_path_angle_deg", path_angle)
            _extend(totals, "direction_wrong_way", (endpoint_angle > 90.0).float())
            _extend(
                totals,
                "direction_velocity_angle_deg",
                velocity_angle[valid_velocity],
            )
            _extend(
                totals,
                "intent_to_motion_gate",
                outputs["intent_to_motion_gate"],
            )
            for index, axis in enumerate(AXES):
                _extend(
                    totals,
                    f"direction_{axis}_sign_accuracy",
                    (classes[:, index] == labels[:, index]).float(),
                )
                _extend(
                    totals,
                    f"direction_{axis}_endpoint_mae_cm",
                    100.0 * (predicted[:, index] - target[:, index]).abs(),
                )
                predicted_axes[index].extend(predicted[:, index].cpu().tolist())
                true_axes[index].extend(target[:, index].cpu().tolist())

    metrics = {
        name: float(np.mean(values))
        for name, values in totals.items()
        if values
    }
    for index, axis in enumerate(AXES):
        predicted = np.asarray(predicted_axes[index], dtype=np.float64)
        target = np.asarray(true_axes[index], dtype=np.float64)
        if (
            len(predicted) > 1
            and float(predicted.std()) > 1e-9
            and float(target.std()) > 1e-9
        ):
            correlation = float(np.corrcoef(predicted, target)[0, 1])
        else:
            correlation = float("nan")
        metrics[f"direction_{axis}_correlation"] = correlation
    return metrics


@torch.no_grad()
def evaluate(
    model: DirectionAwareCompleteReachModel,
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
    metrics.update(evaluate_direction(
        model,
        loader,
        config,
        context_samples,
        patch_length,
        evaluation_leads,
        canvas_tensor,
        device,
    ))
    settings = config["model"]["direction_aware_complete_reach"]
    metrics["direction_selection_score"] = (
        metrics.get("student_px", float("inf"))
        + float(settings.get("selection_angle_px_per_degree", 0.50))
        * metrics.get("direction_endpoint_angle_deg", float("inf"))
        + float(settings.get("selection_endpoint_px_per_cm", 2.0))
        * metrics.get("complete_endpoint_3d_cm", float("inf"))
    )
    return metrics


def _has_option(name: str) -> bool:
    return any(
        value == name or value.startswith(f"{name}=") for value in sys.argv[1:]
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
            "--config", "configs/tracked_direction_aware_complete_reach.yaml"
        ]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = [
            "--output-dir", "runs/direction_aware_complete_reach"
        ]

    base.make_distillation_window = complete.make_complete_reach_window
    base.milliseconds_to_samples = complete.milliseconds_to_samples
    base.evaluate = evaluate
    base.train_teacher = bridge.train_teacher
    channel.ChannelHorizonLatentDistillationModel = (
        DirectionAwareCompleteReachModel
    )
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    print(
        "direction-aware complete reach: gated EMG intent -> 3D motion + "
        "endpoint/path/axis/velocity direction losses; VIVE remains labels only"
    )
    channel.main()

    output = Path(_option_value(
        "--output-dir", "runs/direction_aware_complete_reach"
    ))
    results_path = output / "results.json"
    if not results_path.exists():
        return
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("config", {})
    test = payload.get("test", {})
    rate = base.effective_rate(config)
    leads = tuple(dict.fromkeys(
        complete.milliseconds_to_samples(value, rate)
        for value in config.get("distillation", {}).get(
            "evaluation_leads_ms", [0, 50, 100, 200, 300, 400]
        )
    ))
    print("\n=== direction-aware complete-reach diagnostics ===")
    print(
        f"  endpoint direction angle: "
        f"{test.get('direction_endpoint_angle_deg', float('nan')):.1f}° | "
        f"wrong-way fraction: "
        f"{100.0 * test.get('direction_wrong_way', float('nan')):.1f}% | "
        f"velocity angle: "
        f"{test.get('direction_velocity_angle_deg', float('nan')):.1f}°"
    )
    for axis in AXES:
        print(
            f"  {axis}: sign="
            f"{100.0 * test.get(f'direction_{axis}_sign_accuracy', float('nan')):5.1f}% "
            f"MAE={test.get(f'direction_{axis}_endpoint_mae_cm', float('nan')):5.2f}cm "
            f"r={test.get(f'direction_{axis}_correlation', float('nan')):+.3f}"
        )
    print("  endpoint angle as history grows")
    for lead in reversed(leads):
        label = int(round(1000.0 * lead / rate))
        print(
            f"    {label:3d}ms: "
            f"{test.get(f'direction_endpoint_angle_{label}ms_deg', float('nan')):6.1f}°"
        )
    print(
        f"  learned EMG-intent -> 3D-motion gate: "
        f"{test.get('intent_to_motion_gate', float('nan')):.3f}"
    )


if __name__ == "__main__":
    main()
