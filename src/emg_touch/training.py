from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .metrics import merge_metric_batches
from .utils import move_batch_to_device


def forward_model(
    model: nn.Module,
    batch: dict[str, Any],
    kind: str,
    include_target: bool = False,
    drop_imu: bool = False,
) -> dict[str, Any]:
    if kind == "teacher":
        return model(
            emg=batch["emg"],
            emg_mask=batch["emg_mask"],
            imu=batch["imu"],
            imu_mask=batch["imu_mask"],
            target=batch["target"] if include_target else None,
            drop_imu=drop_imu,
        )
    if kind == "student":
        return model(
            emg=batch["emg"],
            emg_mask=batch["emg_mask"],
            target=batch["target"] if include_target else None,
        )
    return model(emg=batch["emg"], emg_mask=batch["emg_mask"])


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    kind: str,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    metric_batches = []
    records: list[dict[str, Any]] = []
    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        outputs = forward_model(model, device_batch, kind, include_target=False)
        prediction = outputs["prediction"].detach().cpu()
        target = batch["target"].cpu()
        canvas = batch["canvas_size"].cpu()
        button = batch["button_size"].cpu()
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
            records.append(
                {
                    "trial_id": trial_id,
                    "subject": batch["subject"][index],
                    "configuration": batch["configuration"][index],
                    "cutoff_s": float(batch["cutoff_s"][index]),
                    "target_x": float(target[index, 0]),
                    "target_y": float(target[index, 1]),
                    "prediction_x": float(prediction[index, 0]),
                    "prediction_y": float(prediction[index, 1]),
                    "pixel_error": float(pixel_error[index]),
                    "inside_target_box": bool(inside[index]),
                }
            )
    if not metric_batches:
        raise ValueError("Evaluation loader is empty")
    return merge_metric_batches(metric_batches), records


def optimizer_for(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )


def backward_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    gradient_clip_norm: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    scaler.step(optimizer)
    scaler.update()


def validation_huber(
    model: nn.Module, loader: DataLoader, kind: str, device: torch.device
) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = forward_model(model, batch, kind, include_target=False)
            loss = F.smooth_l1_loss(outputs["prediction"], batch["target"], reduction="sum")
            total += float(loss)
            count += batch["target"].size(0)
    return total / max(count, 1)
