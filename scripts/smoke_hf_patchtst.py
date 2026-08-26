#!/usr/bin/env python3
"""Run one non-persistent optimization step for every exact PatchTST modality."""
from __future__ import annotations

import argparse
import time

import torch

from emg_touch.config import load_config
from emg_touch.data.grid_trajectory import build_grid_trajectory_loaders
from emg_touch.grid_training import grid_point_loss, make_continual_training_batch
from emg_touch.models.hf_patchtst import (
    HF_PATCHTST_MODEL_KINDS,
    build_hf_patchtst_model,
)
from emg_touch.utils import choose_device, move_batch_to_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hf_patchtst_exact.yaml")
    parser.add_argument("--split", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--trajectories", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = choose_device(args.device)
    train_loader, _, _ = build_grid_trajectory_loaders(
        config, args.split, args.scaler
    )
    source_batch = next(iter(train_loader))
    # Limit the smoke step without changing the production batch size.
    source_count = len(source_batch["lengths"])
    retained = min(max(1, args.trajectories), source_count)
    for key, value in list(source_batch.items()):
        if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == source_count:
            source_batch[key] = value[:retained]
        elif isinstance(value, list) and len(value) == source_count:
            source_batch[key] = value[:retained]
    batch = make_continual_training_batch(source_batch, config)

    for kind in HF_PATCHTST_MODEL_KINDS:
        model = build_hf_patchtst_model(kind, config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        device_batch = move_batch_to_device(batch, device)
        started = time.perf_counter()
        outputs = model(device_batch)
        losses = grid_point_loss(outputs, device_batch, config)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        elapsed = time.perf_counter() - started
        print(
            f"{kind} loss={float(losses['loss'].detach()):.6f} "
            f"prediction_shape={tuple(outputs['prediction'].shape)} "
            f"elapsed_s={elapsed:.2f} "
            f"backbone={model.patchtst.__class__.__module__}.{model.patchtst.__class__.__name__}"
        )


if __name__ == "__main__":
    main()
