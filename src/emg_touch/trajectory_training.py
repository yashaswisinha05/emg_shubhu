from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .metrics import merge_metric_batches
from .models.full_trajectory import forward_full_trajectory_model
from .utils import move_batch_to_device


def full_trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    beta: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Robust deterministic coordinate loss in normalized screen coordinates."""

    return F.smooth_l1_loss(prediction, target, beta=beta, reduction=reduction)


@torch.no_grad()
def full_trajectory_validation_loss(
    model: nn.Module,
    loader: DataLoader,
    kind: str,
    device: torch.device,
    beta: float,
) -> float:
    model.eval()
    total = 0.0
    coordinate_count = 0
    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        outputs = forward_full_trajectory_model(model, device_batch, kind)
        loss = full_trajectory_loss(
            outputs["prediction"],
            device_batch["target"],
            beta=beta,
            reduction="sum",
        )
        total += float(loss)
        coordinate_count += int(device_batch["target"].numel())
    if coordinate_count == 0:
        raise ValueError("Validation loader is empty")
    return total / coordinate_count


@torch.no_grad()
def evaluate_full_trajectory_model(
    model: nn.Module,
    loader: DataLoader,
    kind: str,
    device: torch.device,
    fold: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate complete trajectories and return aggregate and trial-level results."""

    model.eval()
    metric_batches: list[dict[str, torch.Tensor]] = []
    records: list[dict[str, Any]] = []
    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        outputs = forward_full_trajectory_model(model, device_batch, kind)
        prediction = outputs["prediction"].detach().cpu()
        target = batch["target"].cpu()
        canvas = batch["canvas_size"].cpu()
        button = batch["button_size"].cpu()
        duration = batch["duration_s"].cpu()
        recording_duration = batch["recording_duration_s"].cpu()
        reaction_time = batch["reaction_time_s"].cpu()
        touch_time = batch["touch_time_s"].cpu()
        emg_window_samples = batch["emg_window_samples"].cpu()
        metric_batches.append(
            {
                "prediction": prediction,
                "target": target,
                "canvas_size": canvas,
                "button_size": button,
            }
        )

        pixel_delta = (prediction - target) * canvas
        pixel_error = torch.linalg.vector_norm(pixel_delta, dim=-1)
        inside = (pixel_delta.abs() <= button / 2.0).all(dim=-1)
        for index, trial_id in enumerate(batch["trial_id"]):
            record: dict[str, Any] = {
                "trial_id": trial_id,
                "subject": batch["subject"][index],
                "configuration": batch["configuration"][index],
                "requested_cutoff": batch["temporal_label"][index],
                "model_kind": kind,
                "duration_s": float(duration[index]),
                "recording_duration_s": float(recording_duration[index]),
                "reaction_time_s": float(reaction_time[index]),
                "touch_time_s": float(touch_time[index]),
                "emg_window_samples": int(emg_window_samples[index]),
                "target_x": float(target[index, 0]),
                "target_y": float(target[index, 1]),
                "prediction_x": float(prediction[index, 0]),
                "prediction_y": float(prediction[index, 1]),
                "pixel_error": float(pixel_error[index]),
                "inside_target_box": bool(inside[index]),
            }
            if fold is not None:
                record["fold"] = int(fold)
            records.append(record)

    if not metric_batches:
        raise ValueError("Evaluation loader is empty")
    return merge_metric_batches(metric_batches), records


def full_trajectory_data_report(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> dict[str, Any]:
    """Describe accepted and excluded trajectories for reproducibility."""

    report: dict[str, Any] = {}
    for name, loader in (
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ):
        dataset = loader.dataset
        durations = list(dataset.durations)
        window_samples = np.asarray(dataset.emg_window_sample_counts, dtype=np.int64)
        report[name] = {
            "accepted_trajectories": len(dataset),
            "excluded_trajectories": len(dataset.excluded),
            "minimum_duration_s": min(durations),
            "maximum_duration_s": max(durations),
            "temporal_label": dataset.temporal_label,
            "emg_window_s": list(dataset.emg_window),
            "emg_window_nonempty_fraction": float(np.mean(window_samples > 0)),
            "median_emg_window_samples": float(np.median(window_samples)),
            "minimum_emg_window_samples": int(window_samples.min()),
            "maximum_emg_window_samples": int(window_samples.max()),
            "excluded": dataset.excluded,
        }
    return report
