#!/usr/bin/env python3
"""Evaluate a trained MAHG transfer checkpoint on held-out subjects."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.data.mahg_emg import discover, make_windows, preprocess, subject_for
from scripts.finetune_mahg_emg import make_model


def report(truth, prediction, class_count):
    return {
        "windows": len(truth),
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
        "confusion": confusion_matrix(
            truth, prediction, labels=range(class_count)).tolist(),
    }


@torch.no_grad()
def predict(model, features, targets, batch_size, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(features),
                        torch.from_numpy(targets)), batch_size=batch_size)
    truth, prediction = [], []
    model.eval()
    for values, labels in loader:
        output = model(values.to(device)).argmax(-1).cpu().numpy()
        truth.extend(labels.numpy().tolist())
        prediction.extend(output.tolist())
    return truth, prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+",
                        help="defaults to the checkpoint's untouched test subject")
    parser.add_argument("--files-per-subject", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path,
                        help="optional JSON destination; results always print")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available()
                          else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") != "mahg_emg_transfer_v1":
        parser.error("--checkpoint must be a *_best.pt written by finetune_mahg_emg.py")
    labels = state["labels"]
    label_index = {name: index for index, name in enumerate(labels)}
    files_per_subject = args.files_per_subject or state.get("files_per_subject", 10)
    subjects = args.subjects
    if subjects is None:
        subjects = state.get("subject_split", {}).get("test", [10])
    selected = [path for path in discover(args.root)
                if subject_for(path, files_per_subject) in subjects]
    if not selected:
        parser.error(f"no EMGData files found for subjects {subjects} below {args.root}")

    model = make_model(state["encoder_kind"], state["model_args"], len(labels)).to(device)
    model.load_state_dict(state["state_dict"])
    settings = state["preprocessing"]
    window_steps = round(settings["window_ms"] * settings["output_rate_hz"] / 1000)
    stride_steps = max(1, round(settings.get("stride_ms", 50)
                                * settings["output_rate_hz"] / 1000))
    mean = np.asarray(state["normalization"]["mean"])
    std = np.asarray(state["normalization"]["std"])
    grouped = {}
    for path in selected:
        subject = subject_for(path, files_per_subject)
        features, flags, names, valid = preprocess(
            path, settings["raw_rate_hz"], settings["output_rate_hz"])
        x, names = make_windows(features, flags, names, valid,
                                window_steps, stride_steps)
        if not len(x):
            continue
        unknown = sorted(set(map(str, names)) - set(labels))
        if unknown:
            raise ValueError(f"{path}: labels absent from checkpoint: {unknown}")
        x[..., :8] = (x[..., :8] - mean) / std
        y = np.asarray([label_index[str(name)] for name in names], dtype="int64")
        grouped.setdefault(subject, []).append((x, y))
    if not grouped:
        parser.error("selected recordings produced no valid constant-label windows")

    result = {"checkpoint": str(args.checkpoint), "root": str(args.root),
              "labels": labels, "subjects": subjects, "per_subject": {}}
    all_truth, all_prediction = [], []
    for subject, entries in sorted(grouped.items()):
        x = np.concatenate([entry[0] for entry in entries])
        y = np.concatenate([entry[1] for entry in entries])
        truth, prediction = predict(model, x, y, args.batch_size, device)
        result["per_subject"][str(subject)] = report(truth, prediction, len(labels))
        all_truth.extend(truth); all_prediction.extend(prediction)
    result["overall"] = report(all_truth, all_prediction, len(labels))
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
