#!/usr/bin/env python3
"""Run one non-persistent continual-attention optimization step per model."""
from __future__ import annotations

import argparse

import torch

from emg_touch.config import load_config
from emg_touch.data.grid_trajectory import build_grid_trajectory_loaders
from emg_touch.grid_training import make_continual_training_batch, grid_point_loss
from emg_touch.models.grid_point import GRID_MODEL_KINDS, build_grid_model
from emg_touch.utils import choose_device, move_batch_to_device, seed_everything


def small_batch(batch: dict, examples: int) -> dict:
    original_size = len(batch["trial_id"])
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim and value.size(0) == original_size:
            result[key] = value[:examples]
        elif isinstance(value, list) and len(value) == original_size:
            result[key] = value[:examples]
        else:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/continual_attention.yaml")
    parser.add_argument(
        "--split", default="artifacts/trajectory_cv/mix7/fold-0/split.json"
    )
    parser.add_argument(
        "--scaler", default="artifacts/hybrid_point/mix7/fold-0/scaler.npz"
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--examples", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = choose_device(args.device)
    train_loader, _, _ = build_grid_trajectory_loaders(
        config, args.split, args.scaler
    )
    batch = make_continual_training_batch(next(iter(train_loader)), config)
    batch = move_batch_to_device(small_batch(batch, args.examples), device)

    for kind in GRID_MODEL_KINDS:
        model = build_grid_model(kind, config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        losses = grid_point_loss(outputs, batch, config)
        loss = losses["loss"]
        if device.type == "mps":
            torch.mps.synchronize()
        loss_value = float(loss.detach().cpu())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {kind} smoke-test loss")
        if loss_value <= 1e-8:
            components = {
                name: float(value.detach().cpu()) for name, value in losses.items()
            }
            diagnostics = {
                "loss_weight": batch.get("loss_weight", torch.empty(0, device=device))
                .detach()
                .cpu()
                .tolist(),
                "target": batch["target"].detach().cpu().tolist(),
                "canvas_size": batch["canvas_size"].detach().cpu().tolist(),
                "heatmap_min": float(outputs["heatmap_logits"].detach().min().cpu()),
                "heatmap_max": float(outputs["heatmap_logits"].detach().max().cpu()),
            }
            raise FloatingPointError(
                f"Unexpected zero {kind} smoke-test loss: {components}; "
                f"diagnostics={diagnostics}"
            )
        loss.backward()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        mean_prediction = outputs["prediction"].detach().mean(dim=0).cpu().tolist()
        print(
            f"{kind} {device} step_ok loss={loss_value:.6f} "
            f"mean_prediction={mean_prediction}"
        )
        del model, optimizer
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
