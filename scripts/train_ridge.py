#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

from emg_touch.config import load_config
from emg_touch.data.dataset import TouchTrialDataset
from emg_touch.data.manifest import load_manifest
from emg_touch.data.preprocessing import RobustScaler
from emg_touch.data.splits import subset_from_trial_ids
from emg_touch.metrics import coordinate_metrics
from emg_touch.utils import load_json, save_json, seed_everything


def emg_features(sample: dict) -> np.ndarray:
    values = sample["emg"].numpy()
    mask = sample["emg_mask"].numpy()
    features = []
    for channel in range(values.shape[1]):
        valid = values[mask[:, channel], channel]
        if len(valid) == 0:
            features.extend([0.0] * 10)
            continue
        gradient = np.diff(valid) if len(valid) > 1 else np.zeros(1)
        features.extend(
            [
                float(np.mean(valid)), float(np.std(valid)),
                float(np.sqrt(np.mean(valid**2))), float(np.median(valid)),
                float(np.percentile(valid, 10)), float(np.percentile(valid, 90)),
                float(np.min(valid)), float(np.max(valid)),
                float(np.mean(np.abs(gradient))), float(len(valid) / len(values)),
            ]
        )
    return np.asarray(features, dtype=np.float32)


def materialize(dataset: TouchTrialDataset) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    features, targets, metadata = [], [], []
    for index in tqdm(range(len(dataset))):
        sample = dataset[index]
        features.append(emg_features(sample))
        targets.append(sample["target"].numpy())
        metadata.append(sample)
    return np.stack(features), np.stack(targets), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mean and handcrafted-EMG ridge baselines")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split")
    parser.add_argument("--scaler")
    parser.add_argument("--cutoff", type=float, default=-1.0)
    parser.add_argument("--output-dir", default="runs/ridge")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    frame = load_manifest(config["paths"]["manifest"])
    split = load_json(args.split or config["paths"]["split_file"])
    scaler = RobustScaler.load(args.scaler or config["paths"]["scaler"])
    datasets = {
        name: TouchTrialDataset(
            subset_from_trial_ids(frame, split[name]),
            config["data"], scaler, training=False, fixed_cutoff_s=args.cutoff,
        )
        for name in ("train", "test")
    }
    train_x, train_y, _ = materialize(datasets["train"])
    test_x, test_y, metadata = materialize(datasets["test"])
    model = RidgeCV(alphas=np.logspace(-4, 4, 25)).fit(train_x, train_y)
    ridge_prediction = np.clip(model.predict(test_x), 0.0, 1.0)
    mean_prediction = np.broadcast_to(train_y.mean(axis=0), test_y.shape)
    canvas = torch.stack([sample["canvas_size"] for sample in metadata])
    button = torch.stack([sample["button_size"] for sample in metadata])
    ridge_metrics = coordinate_metrics(
        torch.from_numpy(ridge_prediction), torch.from_numpy(test_y), canvas, button
    )
    mean_metrics = coordinate_metrics(
        torch.from_numpy(mean_prediction.copy()), torch.from_numpy(test_y), canvas, button
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "ridge.joblib")
    save_json({"ridge": ridge_metrics, "mean": mean_metrics}, output_dir / "metrics.json")
    records = []
    for index, sample in enumerate(metadata):
        records.append(
            {
                "trial_id": sample["trial_id"], "subject": sample["subject"],
                "configuration": sample["configuration"],
                "target_x": test_y[index, 0], "target_y": test_y[index, 1],
                "ridge_x": ridge_prediction[index, 0], "ridge_y": ridge_prediction[index, 1],
                "mean_x": mean_prediction[index, 0], "mean_y": mean_prediction[index, 1],
            }
        )
    pd.DataFrame(records).to_csv(output_dir / "predictions.csv", index=False)
    print({"ridge": ridge_metrics, "mean": mean_metrics})


if __name__ == "__main__":
    main()

