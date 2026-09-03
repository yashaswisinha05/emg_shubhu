#!/usr/bin/env python3
"""Train rolling EMG+IMU screen-point and relative-3D prediction heads.

This is an isolated successor to the teacher-bridge experiment.  It retains
the causal temporal EMG+IMU encoder, factor/horizon guidance, privileged VIVE
teacher, decoder-space bridge, channel dropout, and modality interventions.
Only the deployable student decoder changes:

* an intent-owned head predicts the final screen ``(x, y)``;
* a motion-owned, horizon-conditioned head predicts the short future 3D path.

Training samples multiple causal cutoffs per trial.  At inference the same
fixed model can therefore run repeatedly as new EMG+IMU arrives.  VIVE,
screen targets, and true time-to-touch are labels only.

Example:

    python scripts/train_rolling_dual_head_model.py \
      --root "/media/.../emg_imu_vive" \
      --config configs/tracked_rolling_dual_head.yaml \
      --cache-dir artifacts/tracked_cache_posture \
      --session-prefixes dev_a1 dev_a2 dev_a3 dev_a4 \
      --device cuda --teacher-epochs 25 --epochs 50 \
      --lead-window-ms 50 400 \
      --output-dir runs/rolling_dual_head
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
from scripts import train_latent_distillation_model as base  # noqa: E402
from scripts import train_teacher_bridge_model as bridge  # noqa: E402
from emg_touch.models.rolling_dual_head_distillation import (  # noqa: E402
    RollingDualHeadDistillationModel,
)


_BRIDGE_STUDENT_OBJECTIVE = bridge.student_objective
_BRIDGE_EVALUATE = bridge.evaluate


def dual_head_losses(
    outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Directly supervise screen, 3D endpoint, velocity, and curvature."""
    settings = config["model"]["rolling_dual_head"]
    predicted = outputs["trajectory"]
    target = window["trajectory_target"]
    trajectory_scale = max(
        float(config["model"].get("trajectory_limit_m", 0.8)), 1e-6
    )
    pixel_scale = max(float(settings.get("pixel_normalizer_px", 100.0)), 1.0)
    screen_residual = (
        (outputs["prediction"] - window["target"]) * window["canvas_size"]
    ) / pixel_scale
    screen_xy = F.smooth_l1_loss(
        screen_residual,
        torch.zeros_like(screen_residual),
        beta=float(settings.get("screen_huber_beta", 0.1)),
    )
    endpoint = F.smooth_l1_loss(
        predicted[:, -1] / trajectory_scale,
        target[:, -1] / trajectory_scale,
        beta=float(settings.get("motion_huber_beta", 0.05)),
    )
    predicted_velocity = torch.diff(predicted, dim=1)
    target_velocity = torch.diff(target, dim=1)
    velocity = F.smooth_l1_loss(
        predicted_velocity / trajectory_scale,
        target_velocity / trajectory_scale,
        beta=float(settings.get("motion_huber_beta", 0.05)),
    )
    if predicted.size(1) > 2:
        predicted_curve = torch.diff(predicted_velocity, dim=1)
        target_curve = torch.diff(target_velocity, dim=1)
        curvature = F.smooth_l1_loss(
            predicted_curve / trajectory_scale,
            target_curve / trajectory_scale,
            beta=float(settings.get("motion_huber_beta", 0.05)),
        )
    else:
        curvature = predicted.new_zeros(())
    return {
        "dual_screen_xy": screen_xy,
        "dual_trajectory_endpoint": endpoint,
        "dual_trajectory_velocity": velocity,
        "dual_trajectory_curvature": curvature,
    }


def student_objective(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    combined = _BRIDGE_STUDENT_OBJECTIVE(
        outputs, teacher_outputs, window, config
    )
    losses = dual_head_losses(outputs, window, config)
    settings = config["model"]["rolling_dual_head"]
    combined["loss"] = (
        combined["loss"]
        + float(settings.get("screen_xy_weight", 0.5))
        * losses["dual_screen_xy"]
        + float(settings.get("trajectory_endpoint_weight", 0.5))
        * losses["dual_trajectory_endpoint"]
        + float(settings.get("trajectory_velocity_weight", 0.25))
        * losses["dual_trajectory_velocity"]
        + float(settings.get("trajectory_curvature_weight", 0.05))
        * losses["dual_trajectory_curvature"]
    )
    combined.update({name: value.detach() for name, value in losses.items()})
    combined["screen_shared_gate"] = outputs["screen_shared_gate"].mean().detach()
    combined["motion_shared_gate"] = outputs["motion_shared_gate"].mean().detach()
    return combined


def _extend(
    totals: dict[str, list[float]], name: str, values: torch.Tensor
) -> None:
    totals.setdefault(name, []).extend(
        values.detach().cpu().reshape(-1).tolist()
    )


@torch.no_grad()
def evaluate_dual_heads(
    model: RollingDualHeadDistillationModel,
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
            window = base.make_distillation_window(
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
                window["emg"],
                window["imu"],
                window["time_mask"],
                sample=False,
            )
            pixel = (
                (outputs["prediction"] - window["target"])
                * window["canvas_size"]
            ).norm(dim=-1)
            trajectory = 100.0 * (
                outputs["trajectory"] - window["trajectory_target"]
            ).norm(dim=-1).mean(dim=-1)
            endpoint = 100.0 * (
                outputs["trajectory"][:, -1]
                - window["trajectory_target"][:, -1]
            ).norm(dim=-1)
            velocity = 100.0 * (
                torch.diff(outputs["trajectory"], dim=1)
                - torch.diff(window["trajectory_target"], dim=1)
            ).norm(dim=-1).mean(dim=-1)
            label = int(round(1000.0 * lead / rate))
            _extend(totals, f"dual_screen_{label}ms_px", pixel)
            _extend(totals, f"dual_trajectory_{label}ms_cm", trajectory)
            _extend(totals, f"dual_endpoint_{label}ms_cm", endpoint)
            _extend(totals, f"dual_velocity_{label}ms_cm", velocity)
            _extend(totals, "dual_screen_px", pixel)
            _extend(totals, "dual_trajectory_cm", trajectory)
            _extend(totals, "dual_endpoint_cm", endpoint)
            _extend(totals, "screen_shared_gate", outputs["screen_shared_gate"])
            _extend(totals, "motion_shared_gate", outputs["motion_shared_gate"])
    return {
        name: float(np.mean(values))
        for name, values in totals.items()
        if values
    }


@torch.no_grad()
def evaluate(
    model: RollingDualHeadDistillationModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    mean_target: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    metrics = _BRIDGE_EVALUATE(
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
    metrics.update(evaluate_dual_heads(
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
        sys.argv[1:1] = ["--config", "configs/tracked_rolling_dual_head.yaml"]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = ["--output-dir", "runs/rolling_dual_head"]

    channel.ChannelHorizonLatentDistillationModel = (
        RollingDualHeadDistillationModel
    )
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    base.evaluate = evaluate
    base.train_teacher = bridge.train_teacher
    print(
        "rolling dual-head experiment: causal EMG-intent screen head + "
        "IMU-motion/horizon 3D head; VIVE remains supervision only"
    )
    channel.main()

    output = Path(_option_value("--output-dir", "runs/rolling_dual_head"))
    results_path = output / "results.json"
    if not results_path.exists():
        return
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("config", {})
    test = payload.get("test", {})
    rate = base.effective_rate(config)
    leads = tuple(dict.fromkeys(
        base.milliseconds_to_samples(value, rate)
        for value in config.get("distillation", {}).get(
            "evaluation_leads_ms", [50, 100, 200, 300, 400]
        )
    ))
    print("\n=== rolling dual-head diagnostics ===")
    print(
        f"  all leads: screen={test.get('dual_screen_px', float('nan')):.1f}px "
        f"| trajectory={test.get('dual_trajectory_cm', float('nan')):.2f}cm "
        f"| endpoint={test.get('dual_endpoint_cm', float('nan')):.2f}cm"
    )
    print("  convergence as new EMG+IMU arrives")
    for lead in reversed(leads):
        label = int(round(1000.0 * lead / rate))
        print(
            f"    {label:3d}ms to touch: "
            f"screen={test.get(f'dual_screen_{label}ms_px', float('nan')):7.1f}px "
            f"3D={test.get(f'dual_trajectory_{label}ms_cm', float('nan')):6.2f}cm "
            f"end={test.get(f'dual_endpoint_{label}ms_cm', float('nan')):6.2f}cm"
        )
    print(
        f"  learned shared residual gates: screen="
        f"{test.get('screen_shared_gate', float('nan')):.3f}, motion="
        f"{test.get('motion_shared_gate', float('nan')):.3f}"
    )


if __name__ == "__main__":
    main()
