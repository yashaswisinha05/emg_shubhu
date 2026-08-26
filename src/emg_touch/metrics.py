from __future__ import annotations

from typing import Any

import numpy as np
import torch


def coordinate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    canvas_sizes: torch.Tensor,
    button_sizes: torch.Tensor,
) -> dict[str, float]:
    predictions = predictions.detach().cpu().float()
    targets = targets.detach().cpu().float()
    canvas_sizes = canvas_sizes.detach().cpu().float()
    button_sizes = button_sizes.detach().cpu().float()
    normalized_delta = predictions - targets
    pixel_delta = normalized_delta * canvas_sizes
    normalized_distance = torch.linalg.vector_norm(normalized_delta, dim=-1).numpy()
    pixel_distance = torch.linalg.vector_norm(pixel_delta, dim=-1).numpy()
    inside = (pixel_delta.abs() <= button_sizes / 2.0).all(dim=-1).float().numpy()
    return {
        "count": float(len(predictions)),
        "mae_x_norm": float(normalized_delta[:, 0].abs().mean()),
        "mae_y_norm": float(normalized_delta[:, 1].abs().mean()),
        "mean_normalized_error": float(np.mean(normalized_distance)),
        "median_pixel_error": float(np.median(pixel_distance)),
        "mean_pixel_error": float(np.mean(pixel_distance)),
        "p90_pixel_error": float(np.percentile(pixel_distance, 90)),
        "within_target_box": float(np.mean(inside)),
    }


def merge_metric_batches(batches: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    return coordinate_metrics(
        predictions=torch.cat([item["prediction"] for item in batches]),
        targets=torch.cat([item["target"] for item in batches]),
        canvas_sizes=torch.cat([item["canvas_size"] for item in batches]),
        button_sizes=torch.cat([item["button_size"] for item in batches]),
    )

