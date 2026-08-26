#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from emg_touch.config import load_config
from emg_touch.data.full_trajectory import build_full_trajectory_loaders
from emg_touch.metrics import merge_metric_batches
from emg_touch.trajectory_training import full_trajectory_data_report
from emg_touch.utils import load_json, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the training-target mean on a full-trajectory held-out fold"
    )
    parser.add_argument("--config", default="configs/full_trajectory.yaml")
    parser.add_argument("--split", help="Trajectory-CV split JSON")
    parser.add_argument("--scaler", help="Training-fold robust scaler NPZ")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    split_path = str(Path(args.split or config["paths"]["split_file"]).resolve())
    scaler_path = str(Path(args.scaler or config["paths"]["scaler"]).resolve())
    seed_everything(int(config["seed"]))
    train_loader, val_loader, test_loader = build_full_trajectory_loaders(
        config, split_path, scaler_path
    )

    target_prefix = "click" if config["data"].get("target", "click") == "click" else "target"
    target_columns = [f"{target_prefix}_x_norm", f"{target_prefix}_y_norm"]
    training_target = torch.as_tensor(
        train_loader.dataset.frame[target_columns].to_numpy(), dtype=torch.float32
    )
    mean_prediction = training_target.mean(dim=0)
    split = load_json(split_path)
    fold = int(split["fold"]) if "fold" in split else None

    metric_batches: list[dict[str, torch.Tensor]] = []
    records = []
    for batch in test_loader:
        target = batch["target"].cpu()
        canvas = batch["canvas_size"].cpu()
        button = batch["button_size"].cpu()
        duration = batch["duration_s"].cpu()
        prediction = mean_prediction.unsqueeze(0).expand_as(target)
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
                    "requested_cutoff": "full",
                    "model_kind": "mean_baseline",
                    "fold": fold,
                    "duration_s": float(duration[index]),
                    "target_x": float(target[index, 0]),
                    "target_y": float(target[index, 1]),
                    "prediction_x": float(prediction[index, 0]),
                    "prediction_y": float(prediction[index, 1]),
                    "pixel_error": float(pixel_error[index]),
                    "inside_target_box": bool(inside[index]),
                }
            )

    if not metric_batches:
        raise ValueError("Test loader is empty")
    metrics = merge_metric_batches(metric_batches)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_dir / "predictions.csv", index=False)
    save_json(metrics, output_dir / "metrics.json")
    save_json(
        full_trajectory_data_report(train_loader, val_loader, test_loader),
        output_dir / "data_report.json",
    )
    print(f"training_target_mean={mean_prediction.tolist()}")
    print(metrics)
    print(f"Wrote {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
