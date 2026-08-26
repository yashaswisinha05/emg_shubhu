from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from .models.heads import cvae_loss, mdn_negative_log_likelihood


def screen_zone_labels(target: torch.Tensor, grid: list[int]) -> torch.Tensor:
    grid_x, grid_y = int(grid[0]), int(grid[1])
    x = torch.clamp((target[:, 0] * grid_x).long(), 0, grid_x - 1)
    y = torch.clamp((target[:, 1] * grid_y).long(), 0, grid_y - 1)
    return y * grid_x + x


def masked_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
    weights = mask.to(elementwise.dtype)
    return (elementwise * weights).sum() / weights.sum().clamp_min(1.0)


def supervised_objective(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    config: dict[str, Any],
    head_type: str,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    training = config["training"]
    target = batch["target"]
    huber = F.smooth_l1_loss(outputs["prediction"], target)
    details: dict[str, torch.Tensor] = {"huber": huber}
    if head_type == "mdn":
        probabilistic = mdn_negative_log_likelihood(outputs["distribution"], target)
    elif head_type == "cvae":
        probabilistic, cvae_details = cvae_loss(outputs["distribution"], target, kl_beta)
        details.update(cvae_details)
    else:
        probabilistic = huber
    details["probabilistic"] = probabilistic

    zone = F.cross_entropy(
        outputs["zone_logits"], screen_zone_labels(target, config["model"]["screen_grid"])
    )
    future = masked_smooth_l1(
        outputs["future_imu"], batch["future_imu"], batch["future_imu_mask"]
    )
    details["zone"] = zone
    details["future_imu"] = future
    total = (
        float(training["probabilistic_weight"]) * probabilistic
        + float(training["coord_huber_weight"]) * huber
        + float(training["zone_weight"]) * zone
        + float(training["future_imu_weight"]) * future
    )
    details["total"] = total
    return total, {key: float(value.detach()) for key, value in details.items()}


def baseline_objective(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
    return F.smooth_l1_loss(outputs["prediction"], batch["target"])


def distillation_objective(
    student_outputs: dict[str, Any],
    teacher_outputs: dict[str, Any],
    batch: dict[str, Any],
    config: dict[str, Any],
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    supervised, details = supervised_objective(
        student_outputs, batch, config, config["model"]["head"], kl_beta
    )
    prediction = F.smooth_l1_loss(
        student_outputs["prediction"], teacher_outputs["prediction"].detach()
    )
    feature = 1.0 - F.cosine_similarity(
        student_outputs["context"], teacher_outputs["context"].detach(), dim=-1
    ).mean()
    total = (
        supervised
        + float(config["training"]["distill_prediction_weight"]) * prediction
        + float(config["training"]["distill_feature_weight"]) * feature
    )
    details.update(
        {
            "distill_prediction": float(prediction.detach()),
            "distill_feature": float(feature.detach()),
            "total_with_distillation": float(total.detach()),
        }
    )
    return total, details

