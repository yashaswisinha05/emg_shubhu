#!/usr/bin/env python3
"""Train a complete reach whose decoded 3D path cannot reverse per axis.

This is an isolated successor to ``train_direction_aware_complete_reach.py``.
It keeps the same causal EMG+IMU encoder, VAE teacher, guidance losses,
EMG-only branch, and direction supervision, but replaces the unconstrained 3D
trajectory with a hard signed-endpoint and cumulative-positive-progress
decoder.  VIVE position and trajectory are labels only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import train_channel_horizon_distillation_model as channel  # noqa: E402
from scripts import train_complete_reach_model as complete  # noqa: E402
from scripts import train_direction_aware_complete_reach as direction  # noqa: E402
from scripts import train_latent_distillation_model as base  # noqa: E402
from scripts import train_teacher_bridge_model as bridge  # noqa: E402
from emg_touch.models.monotonic_complete_reach import (  # noqa: E402
    MonotonicCompleteReachModel,
)


# Public alias used by tests and makes the retained objective explicit.
student_objective = direction.student_objective
AXES = ("x", "y", "z")


def monotonicity_statistics(
    trajectory: torch.Tensor, tolerance_m: float = 1e-7
) -> dict[str, torch.Tensor]:
    """Measure steps moving away from the trajectory's own endpoint sign."""
    endpoint = trajectory[:, -1]
    delta = torch.diff(trajectory, dim=1)
    # delta * endpoint must be non-negative for every constrained step. Using
    # endpoint rather than sign also treats stationary axes as valid.
    signed_progress = delta * endpoint[:, None, :]
    reverse = signed_progress < -float(tolerance_m)
    result = {
        "monotonic_reverse_step_fraction": reverse.float().mean(dim=(1, 2)),
        "monotonic_any_reverse": reverse.any(dim=2).any(dim=1).float(),
    }
    for index, axis in enumerate(AXES):
        result[f"monotonic_{axis}_reverse_step_fraction"] = (
            reverse[:, :, index].float().mean(dim=1)
        )
    return result


def _extend(
    totals: dict[str, list[float]], name: str, values: torch.Tensor
) -> None:
    totals.setdefault(name, []).extend(
        values.detach().cpu().reshape(-1).tolist()
    )


@torch.no_grad()
def evaluate_monotonicity(
    model: MonotonicCompleteReachModel,
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
    tolerance = float(
        config["model"].get("monotonic_complete_reach", {}).get(
            "reverse_tolerance_m", 1e-7
        )
    )
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
            statistics = monotonicity_statistics(
                outputs["trajectory"], tolerance
            )
            for name, values in statistics.items():
                _extend(totals, name, values)
    return {
        name: float(np.mean(values))
        for name, values in totals.items()
        if values
    }


@torch.no_grad()
def evaluate(
    model: MonotonicCompleteReachModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    mean_target: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    metrics = direction.evaluate(
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
    metrics.update(evaluate_monotonicity(
        model,
        loader,
        config,
        context_samples,
        patch_length,
        evaluation_leads,
        canvas_tensor,
        device,
    ))
    settings = config["model"].get("monotonic_complete_reach", {})
    metrics["monotonic_selection_score"] = (
        metrics.get("direction_selection_score", float("inf"))
        + float(settings.get("selection_path_px_per_cm", 2.0))
        * metrics.get("complete_path_cm", float("inf"))
    )
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
            "--config", "configs/tracked_monotonic_complete_reach.yaml"
        ]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = [
            "--output-dir", "runs/monotonic_complete_reach"
        ]

    base.make_distillation_window = complete.make_complete_reach_window
    base.milliseconds_to_samples = complete.milliseconds_to_samples
    base.evaluate = evaluate
    base.train_teacher = bridge.train_teacher
    channel.ChannelHorizonLatentDistillationModel = MonotonicCompleteReachModel
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    print(
        "hard monotonic complete reach: categorical XYZ sign + positive "
        "cumulative progress; opposite-axis segments are impossible; "
        "VIVE remains labels only"
    )
    channel.main()

    output = Path(_option_value(
        "--output-dir", "runs/monotonic_complete_reach"
    ))
    results_path = output / "results.json"
    if not results_path.exists():
        return
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    test = payload.get("test", {})
    print("\n=== hard monotonic complete-reach diagnostics ===")
    print(
        "  trajectories containing an opposite-axis step: "
        f"{100.0 * test.get('monotonic_any_reverse', float('nan')):.3f}%"
    )
    print(
        "  all-axis opposite-step fraction: "
        f"{100.0 * test.get('monotonic_reverse_step_fraction', float('nan')):.5f}%"
    )
    for axis in AXES:
        print(
            f"  {axis} opposite-step fraction: "
            f"{100.0 * test.get(f'monotonic_{axis}_reverse_step_fraction', float('nan')):.5f}%"
        )
    print(
        "  endpoint/path construction error: "
        f"{test.get('complete_endpoint_consistency_cm', float('nan')):.8f} cm"
    )
    print(
        "  endpoint direction angle: "
        f"{test.get('direction_endpoint_angle_deg', float('nan')):.1f}° | "
        "wrong-way endpoint fraction: "
        f"{100.0 * test.get('direction_wrong_way', float('nan')):.1f}%"
    )


if __name__ == "__main__":
    main()
