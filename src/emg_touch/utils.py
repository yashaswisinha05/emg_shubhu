from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def choose_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    # Asynchronous host transfers require CUDA pinned memory. On MPS they can
    # race for small tensors created during collation (for example continual
    # prefix weights), so keep Apple-device copies synchronous.
    non_blocking = device.type == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * count
        self.count += count

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)
