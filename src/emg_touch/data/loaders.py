from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import TouchTrialDataset
from .manifest import load_manifest
from .preprocessing import RobustScaler
from .splits import subset_from_trial_ids
from ..utils import load_json


def build_loaders(
    config: dict[str, Any],
    split_path: str | None = None,
    scaler_path: str | None = None,
    eval_cutoff_s: float | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    frame = load_manifest(config["paths"]["manifest"])
    split = load_json(split_path or config["paths"]["split_file"])
    scaler = RobustScaler.load(scaler_path or config["paths"]["scaler"])
    datasets = {
        name: TouchTrialDataset(
            subset_from_trial_ids(frame, split[name]),
            config["data"],
            scaler,
            training=name == "train",
            fixed_cutoff_s=eval_cutoff_s if name != "train" else None,
        )
        for name in ("train", "val", "test")
    }
    batch_size = int(config["training"]["batch_size"])
    workers = int(config["training"]["num_workers"])
    loaders = []
    for name in ("train", "val", "test"):
        loaders.append(
            DataLoader(
                datasets[name],
                batch_size=batch_size,
                shuffle=name == "train",
                num_workers=workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=workers > 0,
                drop_last=name == "train",
            )
        )
    return tuple(loaders)  # type: ignore[return-value]
