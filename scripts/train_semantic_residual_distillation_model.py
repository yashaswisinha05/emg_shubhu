#!/usr/bin/env python3
"""Train intent-selective latent distillation with endpoint residuals.

This is an isolated successor to the channel+horizon experiment.  It retains
that model's EMG channel gates, causal horizon latent, factor guidance,
EMG-only objective, and IMU modality dropout.  It additionally:

* aligns only the teacher/student intent blocks with cosine, relational, and
  target-aware cross-modal contrastive losses;
* learns bounded fused and EMG-only endpoint-logit corrections;
* directly supervises those corrections using the true endpoint, blended
  conservatively with the teacher only on examples where the teacher helps.

The teacher sees future VIVE trajectory during training only.  Deployed
``student_forward`` accepts only causal EMG, IMU, and a time mask.

Example:

    python scripts/train_semantic_residual_distillation_model.py \
      --root "/media/.../emg_imu_vive" \
      --config configs/tracked_semantic_residual_distillation.yaml \
      --cache-dir artifacts/tracked_cache_posture \
      --session-prefixes dev_a1 dev_a2 dev_a3 dev_a4 \
      --device cuda --teacher-epochs 25 --epochs 50 --finetune-epochs 0 \
      --lead-window-ms 50 400 \
      --output-dir runs/semantic_residual_distillation
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
from scripts import train_channel_horizon_distillation_model as channel  # noqa: E402
from emg_touch.models.semantic_residual_distillation import (  # noqa: E402
    SemanticResidualDistillationModel,
)


_CHANNEL_STUDENT_OBJECTIVE = channel.student_objective
_BASE_EVALUATE = base.evaluate


def target_aware_contrastive_loss(
    student_intent: torch.Tensor,
    teacher_intent: torch.Tensor,
    target_class: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Cross-modal InfoNCE with all examples at the same target as positives."""
    student = F.normalize(student_intent, dim=-1)
    teacher = F.normalize(teacher_intent.detach(), dim=-1)
    logits = student @ teacher.transpose(0, 1) / max(float(temperature), 1e-3)
    positive = target_class[:, None].eq(target_class[None, :])
    log_probabilities = logits.log_softmax(dim=-1)
    positive_count = positive.sum(dim=-1).clamp_min(1)
    return -(
        log_probabilities * positive.to(log_probabilities.dtype)
    ).sum(dim=-1).div(positive_count).mean()


def relational_intent_loss(
    student_intent: torch.Tensor, teacher_intent: torch.Tensor
) -> torch.Tensor:
    """Preserve the teacher's pairwise semantic geometry, not its nuisance axes."""
    if student_intent.size(0) < 2:
        return student_intent.new_zeros(())
    student_relation = F.normalize(student_intent, dim=-1) @ F.normalize(
        student_intent, dim=-1
    ).transpose(0, 1)
    teacher_normalized = F.normalize(teacher_intent.detach(), dim=-1)
    teacher_relation = teacher_normalized @ teacher_normalized.transpose(0, 1)
    mask = ~torch.eye(
        student_intent.size(0), dtype=torch.bool, device=student_intent.device
    )
    return F.smooth_l1_loss(student_relation[mask], teacher_relation[mask])


def _pixel_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    canvas_size: torch.Tensor,
) -> torch.Tensor:
    return ((prediction - target) * canvas_size).norm(dim=-1)


def _logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1e-4, 1.0 - 1e-4))


def residual_target_loss(
    outputs: dict[str, torch.Tensor],
    teacher_prediction: torch.Tensor,
    target: torch.Tensor,
    canvas_size: torch.Tensor,
    settings: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teach the bounded correction while trusting the oracle selectively."""
    base_logits = outputs["base_direct_logits"].detach()
    truth_delta = _logit(target) - base_logits
    teacher_delta = _logit(teacher_prediction.detach()) - base_logits
    base_error = _pixel_error(
        outputs["base_prediction"].detach(), target, canvas_size
    )
    teacher_error = _pixel_error(teacher_prediction.detach(), target, canvas_size)
    temperature = max(
        float(settings.get("teacher_advantage_temperature_px", 40.0)), 1e-3
    )
    teacher_is_useful = torch.sigmoid((base_error - teacher_error) / temperature)
    teacher_mix = (
        float(settings.get("teacher_target_mix", 0.25)) * teacher_is_useful
    )
    desired_delta = (
        (1.0 - teacher_mix[:, None]) * truth_delta
        + teacher_mix[:, None] * teacher_delta
    )
    maximum = float(settings.get("maximum_logit_delta", 1.5))
    desired_delta = desired_delta.clamp(-maximum, maximum)
    per_sample = F.smooth_l1_loss(
        outputs["residual_logit_delta"], desired_delta, reduction="none"
    ).mean(dim=-1)
    advantage_weight = 1.0 + float(
        settings.get("teacher_advantage_weight", 1.0)
    ) * teacher_is_useful
    return (per_sample * advantage_weight).mean(), teacher_is_useful.mean()


def semantic_distillation_losses(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    settings = config["model"]["semantic_residual"]
    intent_dim = int(config["model"]["factor_latent"]["intent_dim"])
    teacher_intent = teacher_outputs["mu"][:, :intent_dim]
    fused_intent = outputs["mu"][:, :intent_dim]
    emg_intent = outputs["emg_only"]["mu"][:, :intent_dim]
    grid_width, grid_height = map(int, config["model"]["grid_size"])
    target_class = base.target_cell_labels(
        window["target"], grid_width, grid_height
    )
    fused_cosine = (1.0 - F.cosine_similarity(
        fused_intent, teacher_intent.detach(), dim=-1
    )).mean()
    emg_cosine = (1.0 - F.cosine_similarity(
        emg_intent, teacher_intent.detach(), dim=-1
    )).mean()
    temperature = float(settings.get("contrastive_temperature", 0.12))
    fused_contrastive = target_aware_contrastive_loss(
        fused_intent, teacher_intent, target_class, temperature
    )
    emg_contrastive = target_aware_contrastive_loss(
        emg_intent, teacher_intent, target_class, temperature
    )
    fused_relation = relational_intent_loss(fused_intent, teacher_intent)
    emg_relation = relational_intent_loss(emg_intent, teacher_intent)
    residual, teacher_usefulness = residual_target_loss(
        outputs,
        teacher_outputs["prediction"],
        window["target"],
        window["canvas_size"],
        settings,
    )
    emg_residual, _ = residual_target_loss(
        outputs["emg_only"],
        teacher_outputs["prediction"],
        window["target"],
        window["canvas_size"],
        settings,
    )
    return {
        "semantic_fused_cosine": fused_cosine,
        "semantic_emg_cosine": emg_cosine,
        "semantic_fused_contrastive": fused_contrastive,
        "semantic_emg_contrastive": emg_contrastive,
        "semantic_fused_relational": fused_relation,
        "semantic_emg_relational": emg_relation,
        "endpoint_residual": residual,
        "emg_endpoint_residual": emg_residual,
        "teacher_usefulness": teacher_usefulness,
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
    semantic = semantic_distillation_losses(
        outputs, teacher_outputs, window, config
    )
    settings = config["model"]["semantic_residual"]
    combined["loss"] = (
        combined["loss"]
        + float(settings.get("fused_cosine_weight", 0.0))
        * semantic["semantic_fused_cosine"]
        + float(settings.get("emg_cosine_weight", 0.0))
        * semantic["semantic_emg_cosine"]
        + float(settings.get("fused_contrastive_weight", 0.0))
        * semantic["semantic_fused_contrastive"]
        + float(settings.get("emg_contrastive_weight", 0.0))
        * semantic["semantic_emg_contrastive"]
        + float(settings.get("fused_relational_weight", 0.0))
        * semantic["semantic_fused_relational"]
        + float(settings.get("emg_relational_weight", 0.0))
        * semantic["semantic_emg_relational"]
        + float(settings.get("endpoint_residual_weight", 0.0))
        * semantic["endpoint_residual"]
        + float(settings.get("emg_endpoint_residual_weight", 0.0))
        * semantic["emg_endpoint_residual"]
    )
    combined.update({name: value.detach() for name, value in semantic.items()})
    return combined


@torch.no_grad()
def evaluate_semantic_metrics(
    model: SemanticResidualDistillationModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    """Measure whether the residual closes the teacher gap on held-out data."""
    model.eval()
    teacher_steps = int(config["model"]["teacher_trajectory_steps"])
    trajectory_limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    intent_dim = int(config["model"]["factor_latent"]["intent_dim"])
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
                batch, context_samples, patch_length, teacher_steps, generator,
                trajectory_limit, velocity_scale, canvas_tensor, fixed_lead=lead,
            )
            if window is None:
                continue
            teacher = model.teacher_forward(
                window["teacher_features"], sample=False
            )
            student = model.student_forward(
                window["emg"], window["imu"], window["time_mask"],
                sample=False, include_emg_only=True,
            )
            corrected = _pixel_error(
                student["prediction"], window["target"], window["canvas_size"]
            )
            parent = _pixel_error(
                student["base_prediction"], window["target"], window["canvas_size"]
            )
            teacher_error = _pixel_error(
                teacher["prediction"], window["target"], window["canvas_size"]
            )
            emg_corrected = _pixel_error(
                student["emg_only"]["prediction"],
                window["target"], window["canvas_size"],
            )
            emg_parent = _pixel_error(
                student["emg_only"]["base_prediction"],
                window["target"], window["canvas_size"],
            )
            teacher_intent = teacher["mu"][:, :intent_dim]
            for name, value in {
                "base_student_px": parent,
                "residual_gain_px": parent - corrected,
                "teacher_gap_before_px": parent - teacher_error,
                "teacher_gap_after_px": corrected - teacher_error,
                "base_emg_only_px": emg_parent,
                "emg_residual_gain_px": emg_parent - emg_corrected,
                "intent_teacher_cosine": F.cosine_similarity(
                    student["mu"][:, :intent_dim], teacher_intent, dim=-1
                ),
                "emg_intent_teacher_cosine": F.cosine_similarity(
                    student["emg_only"]["mu"][:, :intent_dim],
                    teacher_intent, dim=-1,
                ),
            }.items():
                base._append(totals, name, value)
    return {name: float(np.mean(values)) for name, values in totals.items()}


def evaluate(
    model: SemanticResidualDistillationModel,
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
        model, loader, config, context_samples, patch_length,
        evaluation_leads, canvas_tensor, mean_target, device,
    )
    metrics.update(evaluate_semantic_metrics(
        model, loader, config, context_samples, patch_length,
        evaluation_leads, canvas_tensor, device,
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
            "--config", "configs/tracked_semantic_residual_distillation.yaml"
        ]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = [
            "--output-dir", "runs/semantic_residual_distillation"
        ]
    # Patch only this entrypoint's imported module.  The original scripts and
    # their model files remain unchanged and independently reproducible.
    channel.ChannelHorizonLatentDistillationModel = (
        SemanticResidualDistillationModel
    )
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    base.evaluate = evaluate
    print(
        "semantic residual experiment: selective intent alignment + wearable "
        "endpoint correction; teacher/VIVE remain training-only"
    )
    channel.main()
    results_path = Path(
        _option_value("--output-dir", "runs/semantic_residual_distillation")
    ) / "results.json"
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            test = json.load(handle).get("test", {})
        print("\n=== semantic residual diagnostics ===")
        print(
            f"  parent endpoint before residual : "
            f"{test.get('base_student_px', float('nan')):7.1f} px"
        )
        print(
            f"  learned residual improvement    : "
            f"{test.get('residual_gain_px', float('nan')):+7.1f} px"
        )
        print(
            f"  remaining gap to teacher        : "
            f"{test.get('teacher_gap_after_px', float('nan')):+7.1f} px"
        )
        print(
            f"  EMG-only residual improvement   : "
            f"{test.get('emg_residual_gain_px', float('nan')):+7.1f} px"
        )
        print(
            f"  intent cosine (fused / EMG)     : "
            f"{test.get('intent_teacher_cosine', float('nan')):.3f} / "
            f"{test.get('emg_intent_teacher_cosine', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
