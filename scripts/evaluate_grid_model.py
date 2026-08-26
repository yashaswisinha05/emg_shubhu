#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emg_touch.checkpointing import load_model_state
from emg_touch.config import load_config
from emg_touch.data.grid_trajectory import build_grid_trajectory_loaders
from emg_touch.grid_training import evaluate_grid_model, grid_data_report
from emg_touch.models.grid_point import GRID_MODEL_KINDS, build_grid_model
from emg_touch.utils import choose_device, load_json, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a grid-and-offset checkpoint")
    parser.add_argument("--config", default="configs/grid_point.yaml")
    parser.add_argument("--kind", choices=GRID_MODEL_KINDS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split")
    parser.add_argument("--scaler")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()

    config = load_config(args.config)
    split_path = str(Path(args.split or config["paths"]["split_file"]).resolve())
    scaler_path = str(Path(args.scaler or config["paths"]["scaler"]).resolve())
    seed_everything(int(config["seed"]))
    device = choose_device(args.device)
    train_loader, val_loader, test_loader = build_grid_trajectory_loaders(
        config, split_path, scaler_path
    )
    model = build_grid_model(args.kind, config).to(device)
    checkpoint = load_model_state(model, args.checkpoint)
    if checkpoint.get("model_kind") != args.kind:
        raise ValueError(
            f"Checkpoint kind={checkpoint.get('model_kind')!r}, requested={args.kind!r}"
        )
    split = load_json(split_path)
    fold = int(split["fold"]) if "fold" in split else None
    metrics, records = evaluate_grid_model(
        model, test_loader, args.kind, device, config, fold=fold
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_dir / "predictions.csv", index=False)
    save_json(metrics, output_dir / "metrics.json")
    save_json(
        grid_data_report(train_loader, val_loader, test_loader),
        output_dir / "data_report.json",
    )
    print(metrics)
    print(f"Wrote {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
