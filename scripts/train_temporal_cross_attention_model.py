#!/usr/bin/env python3
"""Train deterministic token-level EMG/IMU intent distillation.

This is a new experiment; it does not modify the older checkpoint families.
The student uses physical-sensor x lag attention, EMG/IMU token
cross-attention, and separately owned intent/motion/residual latent blocks.
Only the teacher sees future VIVE during training.  Deployment still calls
``student_forward(EMG, IMU, time_mask)``.

The final report also divides held-out predictions into equal-count movement-
duration thirds.  This measures whether physically slower trials help; UI
playback speed cannot alter predictions.

Example:

    python scripts/train_temporal_cross_attention_model.py \
      --root "/media/.../emg_imu_vive" \
      --config configs/tracked_temporal_cross_attention.yaml \
      --cache-dir artifacts/tracked_cache_posture \
      --session-prefixes dev_a1 dev_a2 dev_a3 dev_a4 \
      --device cuda --teacher-epochs 25 --epochs 50 \
      --lead-window-ms 50 400 \
      --output-dir runs/temporal_cross_attention
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

from scripts import train_latent_distillation_model as base  # noqa: E402
from scripts import train_semantic_residual_distillation_model as semantic  # noqa: E402
from emg_touch.data.schema import sensor_names  # noqa: E402
from emg_touch.models.temporal_cross_attention_distillation import (  # noqa: E402
    TemporalCrossAttentionDistillationModel,
)


_SEMANTIC_STUDENT_OBJECTIVE = semantic.student_objective
_SEMANTIC_EVALUATE = semantic.evaluate


def lag_attention_entropy(attention: torch.Tensor) -> torch.Tensor:
    """Normalized entropy over all physical sensor x lag choices."""
    flat = attention.flatten(1).clamp_min(1e-8)
    denominator = max(float(np.log(flat.size(1))), 1e-8)
    return -(flat * flat.log()).sum(dim=-1).mean() / denominator


def student_objective(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    combined = _SEMANTIC_STUDENT_OBJECTIVE(
        outputs, teacher_outputs, window, config
    )
    settings = config["model"]["temporal_cross_attention"]
    intent_dim = int(config["model"]["factor_latent"]["intent_dim"])
    teacher_intent = teacher_outputs["mu"][:, :intent_dim].detach()
    fused_alignment = F.smooth_l1_loss(
        outputs["mu"][:, :intent_dim], teacher_intent
    )
    emg_alignment = F.smooth_l1_loss(
        outputs["emg_only"]["mu"][:, :intent_dim], teacher_intent
    )
    lag_entropy = lag_attention_entropy(outputs["lag_attention"])
    combined["loss"] = (
        combined["loss"]
        + float(settings.get("fused_intent_alignment_weight", 0.5))
        * fused_alignment
        + float(settings.get("emg_intent_alignment_weight", 1.0))
        * emg_alignment
        + float(settings.get("lag_entropy_weight", 0.0)) * lag_entropy
    )
    combined.update({
        "deterministic_fused_intent": fused_alignment.detach(),
        "deterministic_emg_intent": emg_alignment.detach(),
        "lag_attention_entropy": lag_entropy.detach(),
    })
    return combined


def duration_bin_summary(
    durations_ms: np.ndarray, errors_px: np.ndarray
) -> dict[str, float]:
    """Equal-count relative duration bins, robust to repeated durations."""
    if len(durations_ms) == 0 or len(durations_ms) != len(errors_px):
        return {}
    order = np.argsort(durations_ms, kind="stable")
    groups = np.array_split(order, 3)
    names = ("fast", "medium", "slow")
    metrics: dict[str, float] = {}
    for name, indices in zip(names, groups):
        if len(indices) == 0:
            continue
        metrics[f"duration_{name}_mean_ms"] = float(durations_ms[indices].mean())
        metrics[f"duration_{name}_student_px"] = float(errors_px[indices].mean())
        metrics[f"duration_{name}_count"] = float(len(indices))
    if all(f"duration_{name}_student_px" in metrics for name in ("fast", "slow")):
        metrics["duration_slow_minus_fast_px"] = (
            metrics["duration_slow_student_px"]
            - metrics["duration_fast_student_px"]
        )
    return metrics


@torch.no_grad()
def evaluate_temporal_architecture(
    model: TemporalCrossAttentionDistillationModel,
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
    teacher_steps = int(config["model"]["teacher_trajectory_steps"])
    trajectory_limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    sensors = list(sensor_names(config["data"]))
    lag_edges = list(
        map(float, config["model"]["temporal_cross_attention"]["lag_edges_ms"])
    )
    durations: list[float] = []
    errors: list[float] = []
    lag_totals: dict[str, list[float]] = {}
    generator = np.random.default_rng(0)
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
                teacher_steps,
                generator,
                trajectory_limit,
                velocity_scale,
                canvas_tensor,
                fixed_lead=lead,
            )
            if window is None:
                continue
            outputs = model.student_forward(
                window["emg"], window["imu"], window["time_mask"],
                sample=False,
            )
            pixel = (
                (outputs["prediction"] - window["target"])
                * window["canvas_size"]
            ).norm(dim=-1)
            duration = 1000.0 * (
                window["samples_past_onset"] + window["lead_samples"]
            ).to(torch.float32) / rate
            durations.extend(duration.cpu().tolist())
            errors.extend(pixel.cpu().tolist())
            attention = outputs["lag_attention"]
            for sensor_index, sensor in enumerate(sensors):
                for lag_index, (low, high) in enumerate(
                    zip(lag_edges[:-1], lag_edges[1:])
                ):
                    key = f"lag_attention_{sensor}_{int(low)}_{int(high)}ms"
                    lag_totals.setdefault(key, []).extend(
                        attention[:, sensor_index, lag_index].cpu().tolist()
                    )
    metrics = duration_bin_summary(
        np.asarray(durations, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
    )
    metrics.update({
        key: float(np.mean(values)) for key, values in lag_totals.items() if values
    })
    return metrics


@torch.no_grad()
def evaluate(
    model: TemporalCrossAttentionDistillationModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    mean_target: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    metrics = _SEMANTIC_EVALUATE(
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
    metrics.update(evaluate_temporal_architecture(
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
        argument == name or argument.startswith(f"{name}=")
        for argument in sys.argv[1:]
    )


def _option_value(name: str, default: str) -> str:
    for index, argument in enumerate(sys.argv[1:]):
        if argument.startswith(f"{name}="):
            return argument.split("=", 1)[1]
        if argument == name and index + 2 <= len(sys.argv[1:]):
            return sys.argv[index + 2]
    return default


def main() -> None:
    if not _has_option("--config"):
        sys.argv[1:1] = [
            "--config", "configs/tracked_temporal_cross_attention.yaml"
        ]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = [
            "--output-dir", "runs/temporal_cross_attention"
        ]
    # Patch only this new entrypoint's imported modules. All old commands and
    # checkpoint architectures remain independently runnable.
    semantic.SemanticResidualDistillationModel = (
        TemporalCrossAttentionDistillationModel
    )
    semantic.student_objective = student_objective
    semantic.evaluate = evaluate
    semantic.__doc__ = __doc__
    print(
        "temporal cross-attention experiment: deterministic EMG intent + "
        "IMU motion + fused residual; VIVE remains teacher/label only"
    )
    semantic.main()

    output = Path(_option_value("--output-dir", "runs/temporal_cross_attention"))
    results_path = output / "results.json"
    if not results_path.exists():
        return
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    test = payload.get("test", {})
    print("\n=== temporal cross-attention diagnostics ===")
    print("  error by physical movement-duration third")
    for name in ("fast", "medium", "slow"):
        print(
            f"    {name:6}: "
            f"{test.get(f'duration_{name}_student_px', float('nan')):7.1f} px "
            f"at mean {test.get(f'duration_{name}_mean_ms', float('nan')):7.0f} ms"
        )
    difference = test.get("duration_slow_minus_fast_px", float("nan"))
    interpretation = "slower helped" if difference < 0 else "slower did not help"
    print(f"    slow-fast: {difference:+.1f} px ({interpretation})")
    print("\n  learned EMG sensor x causal-lag share")
    settings = payload["config"]
    sensors = list(sensor_names(settings["data"]))
    edges = settings["model"]["temporal_cross_attention"]["lag_edges_ms"]
    for low, high in zip(edges[:-1], edges[1:]):
        shares = [
            f"{sensor}={100.0 * test.get(f'lag_attention_{sensor}_{int(low)}_{int(high)}ms', float('nan')):4.1f}%"
            for sensor in sensors
        ]
        print(f"    {int(low):4}-{int(high):4} ms: " + "  ".join(shares))


if __name__ == "__main__":
    main()
