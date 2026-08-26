#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emg_touch.checkpointing import load_model_state
from emg_touch.config import load_config
from emg_touch.data.full_trajectory import build_full_trajectory_loaders
from emg_touch.models.full_trajectory import build_full_trajectory_model
from emg_touch.trajectory_training import (
    evaluate_full_trajectory_model,
    full_trajectory_data_report,
)
from emg_touch.utils import choose_device, load_json, save_json, seed_everything


MODEL_KINDS = ["emg_tcn", "emg_patch", "imu_patch", "emg_residual", "multimodal"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a full-trajectory checkpoint on its held-out fold"
    )
    parser.add_argument("--config", default="configs/full_trajectory.yaml")
    parser.add_argument("--kind", choices=MODEL_KINDS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", help="Trajectory-CV split JSON")
    parser.add_argument("--scaler", help="Training-fold robust scaler NPZ")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", help="cpu, mps, cuda, or a CUDA device such as cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    split_path = str(Path(args.split or config["paths"]["split_file"]).resolve())
    scaler_path = str(Path(args.scaler or config["paths"]["scaler"]).resolve())
    seed_everything(int(config["seed"]))
    device = choose_device(args.device)
    train_loader, val_loader, test_loader = build_full_trajectory_loaders(
        config, split_path, scaler_path
    )
    model = build_full_trajectory_model(args.kind, config).to(device)
    checkpoint = load_model_state(model, args.checkpoint)
    checkpoint_kind = checkpoint.get("model_kind")
    if checkpoint_kind is not None and checkpoint_kind != args.kind:
        raise ValueError(
            f"Checkpoint contains model_kind={checkpoint_kind!r}, not {args.kind!r}"
        )

    split = load_json(split_path)
    fold = int(split["fold"]) if "fold" in split else None
    metrics, records = evaluate_full_trajectory_model(
        model, test_loader, args.kind, device, fold=fold
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_dir / "predictions.csv", index=False)
    save_json(metrics, output_dir / "metrics.json")
    save_json(
        full_trajectory_data_report(train_loader, val_loader, test_loader),
        output_dir / "data_report.json",
    )
    print(metrics)
    print(f"Wrote {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
