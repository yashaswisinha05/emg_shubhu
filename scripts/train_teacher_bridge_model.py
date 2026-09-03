#!/usr/bin/env python3
"""Train wearable factors against teacher decoder space and output hierarchy.

This experiment addresses the measured 80.6 px teacher gap without changing
the teacher's privileged training boundary.  It keeps temporal cross-attention
and factor ownership, but gives the student a decoder copied from the best
teacher and then adapted independently.  Supervision matches teacher
heatmaps, offsets, direct logits, and decoder latents while retaining the true
endpoint loss so teacher mistakes are not copied blindly.

Example:

    python scripts/train_teacher_bridge_model.py \
      --root "/media/.../emg_imu_vive" \
      --config configs/tracked_teacher_bridge.yaml \
      --cache-dir artifacts/tracked_cache_posture \
      --session-prefixes dev_a1 dev_a2 dev_a3 dev_a4 \
      --device cuda --teacher-epochs 25 --epochs 50 \
      --lead-window-ms 50 400 \
      --output-dir runs/teacher_bridge
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
from scripts.train_temporal_cross_attention_model import (  # noqa: E402
    evaluate_temporal_architecture,
)
from emg_touch.models.teacher_bridge_distillation import (  # noqa: E402
    TeacherBridgeDistillationModel,
)


_CHANNEL_STUDENT_OBJECTIVE = channel.student_objective
_BASE_EVALUATE = base.evaluate
_BASE_TRAIN_TEACHER = base.train_teacher


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def hierarchical_teacher_losses(
    outputs: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    window: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Match the teacher's endpoint hierarchy where the teacher is useful."""
    temperature = max(float(settings.get("temperature", 2.0)), 1e-3)
    teacher_probabilities = torch.softmax(
        teacher["heatmap_logits"].detach() / temperature, dim=-1
    )
    student_log_probabilities = torch.log_softmax(
        outputs["heatmap_logits"] / temperature, dim=-1
    )
    teacher_log_probabilities = torch.log_softmax(
        teacher["heatmap_logits"].detach() / temperature, dim=-1
    )
    heatmap = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=-1) * temperature**2

    offset_per_cell = F.smooth_l1_loss(
        outputs["offset_logits"],
        teacher["offset_logits"].detach(),
        reduction="none",
    ).mean(dim=-1)
    offset = (offset_per_cell * teacher_probabilities).sum(dim=-1)
    direct = F.smooth_l1_loss(
        outputs["direct_logits"],
        teacher["direct_logits"].detach(),
        reduction="none",
    ).mean(dim=-1)
    latent = F.smooth_l1_loss(
        outputs["decoder_latent"], teacher["mu"].detach(), reduction="none"
    ).mean(dim=-1)

    student_error = (
        (outputs["prediction"].detach() - window["target"])
        * window["canvas_size"]
    ).norm(dim=-1)
    teacher_error = (
        (teacher["prediction"].detach() - window["target"])
        * window["canvas_size"]
    ).norm(dim=-1)
    scale = max(float(settings.get("advantage_temperature_px", 40.0)), 1e-3)
    teacher_usefulness = torch.sigmoid((student_error - teacher_error) / scale)
    # Retain some output alignment everywhere, but concentrate it on examples
    # where the oracle is demonstrably better than the current student.
    weights = float(settings.get("minimum_teacher_weight", 0.25)) + teacher_usefulness
    return {
        "teacher_heatmap": _weighted_mean(heatmap, weights),
        "teacher_offset": _weighted_mean(offset, weights),
        "teacher_direct": _weighted_mean(direct, weights),
        "teacher_decoder_latent": _weighted_mean(latent, weights),
        "teacher_usefulness": teacher_usefulness.mean(),
    }


def student_objective(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    combined = _CHANNEL_STUDENT_OBJECTIVE(
        outputs, teacher_outputs, window, config
    )
    settings = config["model"]["teacher_bridge"]
    fused = hierarchical_teacher_losses(
        outputs, teacher_outputs, window, settings
    )
    emg = hierarchical_teacher_losses(
        outputs["emg_only"], teacher_outputs, window, settings
    )
    output_loss = (
        float(settings.get("heatmap_weight", 1.0)) * fused["teacher_heatmap"]
        + float(settings.get("offset_weight", 0.5)) * fused["teacher_offset"]
        + float(settings.get("direct_weight", 1.0)) * fused["teacher_direct"]
        + float(settings.get("decoder_latent_weight", 0.25))
        * fused["teacher_decoder_latent"]
    )
    emg_output_loss = (
        float(settings.get("heatmap_weight", 1.0)) * emg["teacher_heatmap"]
        + float(settings.get("offset_weight", 0.5)) * emg["teacher_offset"]
        + float(settings.get("direct_weight", 1.0)) * emg["teacher_direct"]
        + float(settings.get("decoder_latent_weight", 0.25))
        * emg["teacher_decoder_latent"]
    )
    combined["loss"] = (
        combined["loss"]
        + float(settings.get("output_distillation_weight", 1.0)) * output_loss
        + float(settings.get("emg_output_distillation_weight", 0.5))
        * emg_output_loss
    )
    # The inherited Gaussian KL is deliberately zero-weighted for this
    # deterministic student. Preserve it under an explicit diagnostic name
    # and make the standard console ``latent=`` value report the bridge loss.
    combined["disabled_gaussian_kl"] = combined["latent"]
    combined["latent"] = fused["teacher_decoder_latent"].detach()
    combined.update({name: value.detach() for name, value in fused.items()})
    combined.update({
        f"emg_{name}": value.detach() for name, value in emg.items()
    })
    return combined


def train_teacher(*args: Any, **kwargs: Any) -> list[dict[str, float]]:
    """Train the oracle normally, then initialise the independent student head."""
    history = _BASE_TRAIN_TEACHER(*args, **kwargs)
    model = args[0] if args else kwargs["model"]
    if not isinstance(model, TeacherBridgeDistillationModel):
        raise TypeError("teacher-bridge trainer received the wrong model class")
    model.initialise_student_decoder_from_teacher()
    return history


@torch.no_grad()
def evaluate_bridge(
    model: TeacherBridgeDistillationModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    teacher_steps = int(config["model"]["teacher_trajectory_steps"])
    trajectory_limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    rate = base.effective_rate(config)
    generator = np.random.default_rng(0)
    disagreement: list[float] = []
    latent_rmse: list[float] = []
    teacher_better: list[float] = []
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
            teacher = model.teacher_forward(window["teacher_features"], sample=False)
            student = model.student_forward(
                window["emg"], window["imu"], window["time_mask"], sample=False
            )
            teacher_error = (
                (teacher["prediction"] - window["target"])
                * window["canvas_size"]
            ).norm(dim=-1)
            student_error = (
                (student["prediction"] - window["target"])
                * window["canvas_size"]
            ).norm(dim=-1)
            disagreement.extend((
                (student["prediction"] - teacher["prediction"])
                * window["canvas_size"]
            ).norm(dim=-1).cpu().tolist())
            latent_rmse.extend(torch.sqrt(
                (student["decoder_latent"] - teacher["mu"]).square().mean(dim=-1)
            ).cpu().tolist())
            teacher_better.extend((teacher_error < student_error).float().cpu().tolist())
    metrics = {
        "teacher_student_disagreement_px": float(np.mean(disagreement)),
        "decoder_latent_teacher_rmse": float(np.mean(latent_rmse)),
        "teacher_better_fraction": float(np.mean(teacher_better)),
    }
    # Reuse the duration and learned sensor-lag report from the parent model.
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


@torch.no_grad()
def evaluate(
    model: TeacherBridgeDistillationModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    mean_target: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    metrics = _BASE_EVALUATE(
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
    metrics.update(evaluate_bridge(
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
        sys.argv[1:1] = ["--config", "configs/tracked_teacher_bridge.yaml"]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = ["--output-dir", "runs/teacher_bridge"]
    channel.ChannelHorizonLatentDistillationModel = TeacherBridgeDistillationModel
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    base.evaluate = evaluate
    base.train_teacher = train_teacher
    print(
        "teacher bridge experiment: independent student decoder + hierarchical "
        "teacher-output distillation; deployment remains EMG+IMU only"
    )
    channel.main()

    output = Path(_option_value("--output-dir", "runs/teacher_bridge"))
    results_path = output / "results.json"
    if not results_path.exists():
        return
    with results_path.open("r", encoding="utf-8") as handle:
        test = json.load(handle).get("test", {})
    print("\n=== teacher bridge diagnostics ===")
    print(
        f"  student-teacher endpoint disagreement: "
        f"{test.get('teacher_student_disagreement_px', float('nan')):.1f} px"
    )
    print(
        f"  decoder-latent teacher RMSE          : "
        f"{test.get('decoder_latent_teacher_rmse', float('nan')):.4f}"
    )
    print(
        f"  teacher better than student          : "
        f"{100.0 * test.get('teacher_better_fraction', float('nan')):.1f}%"
    )
    print(
        f"  remaining endpoint gap               : "
        f"{test.get('student_px', float('nan')) - test.get('teacher_px', float('nan')):+.1f} px"
    )
    slow_difference = test.get("duration_slow_minus_fast_px", float("nan"))
    print(f"  slow-fast movement-duration effect   : {slow_difference:+.1f} px")


if __name__ == "__main__":
    main()
