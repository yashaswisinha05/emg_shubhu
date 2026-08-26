from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    model_kind: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_config = {key: value for key, value in config.items() if not key.startswith("_")}
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "config": clean_config,
            "metrics": metrics,
            "model_kind": model_kind,
        },
        path,
    )


def load_model_state(model: nn.Module, path: str | Path, strict: bool = True) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=strict)
    return checkpoint
