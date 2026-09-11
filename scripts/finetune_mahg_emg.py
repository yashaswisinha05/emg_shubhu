#!/usr/bin/env python3
"""Subject-held-out MAHG-EMG transfer for GRU or neuromuscular EMG encoders."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from emg_touch.data.mahg_emg import discover, make_windows, preprocess, subject_for
from emg_touch.models.reach_grasp_architecture_baselines import ArchitectureBaseline
from emg_touch.models.reach_grasp_neuromuscular_future import (
    NeuromuscularFutureGripperPoseModel,
)
from emg_touch.models.reach_grasp_patch_transformer import CausalPatchBranch


class TransferGRU(nn.Module):
    def __init__(self, width=128, layers=4, dropout=.1, classes=5):
        super().__init__()
        self.width, self.layers = width, layers
        self.input = nn.Sequential(nn.Linear(64, width), nn.LayerNorm(width), nn.GELU())
        self.recurrent = nn.GRU(width, width, layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0)
        self.head = nn.Linear(width, classes)

    def forward(self, emg):
        imu = emg.new_zeros((*emg.shape[:2], 48))
        self.recurrent.flatten_parameters()
        sequence = self.recurrent(self.input(torch.cat((emg, imu), -1)))[0]
        return self.head(sequence[:, -1])


class TransferPatch(nn.Module):
    """MAHG head over the pretrained neuromuscular model's EMG encoder only."""

    def __init__(self, width=128, patch=16, stride=4, layers=4, heads=4,
                 dropout=.1, classes=5):
        super().__init__()
        self.layers = layers
        self.encoder = CausalPatchBranch(16, width, patch, stride, layers, heads, dropout)
        self.head = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, width),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(width, classes))

    def forward(self, emg):
        features = self.encoder.forward_features(emg)
        return self.head(torch.cat((features["context"][:, -1],
                                    features["local"][:, -1]), -1))


def source_spec(checkpoint):
    args = checkpoint.get("model_args", {})
    if (checkpoint.get("format") == "reach_grasp_architecture_baseline_v1"
            and checkpoint.get("architecture") == "gru"):
        return "gru", {key: args[key] for key in ("width", "layers", "dropout")}
    if checkpoint.get("format") == "gripper_neuromuscular_future_v1":
        keys = ("width", "patch", "stride", "layers", "heads", "dropout")
        return "neuromuscular_patch", {key: args[key] for key in keys}
    raise ValueError("checkpoint must be a causal-GRU architecture baseline or "
                     "gripper_neuromuscular_future_v1 checkpoint")


def make_model(kind, model_args, classes):
    constructor = TransferGRU if kind == "gru" else TransferPatch
    return constructor(**model_args, classes=classes)


def initialize_from_checkpoint(model, checkpoint, kind):
    args = checkpoint["model_args"]
    if kind == "gru":
        source = ArchitectureBaseline("gru", **args)
        source.load_state_dict(checkpoint["state_dict"])
        model.input.load_state_dict(source.input.state_dict())
        model.recurrent.load_state_dict(source.recurrent.state_dict())
    else:
        source = NeuromuscularFutureGripperPoseModel(**args)
        source.load_state_dict(checkpoint["state_dict"])
        model.encoder.load_state_dict(source.emg.state_dict())


def set_trainable(model, protocol):
    for parameter in model.parameters():
        parameter.requires_grad = protocol == "scratch"
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    if protocol == "finetune":
        if isinstance(model, TransferGRU):
            suffix = f"_l{model.layers - 1}"
            for name, parameter in model.recurrent.named_parameters():
                parameter.requires_grad = name.endswith(suffix)
        else:
            for parameter in model.encoder.transformer.layers[-1].parameters():
                parameter.requires_grad = True
            for parameter in model.encoder.output.parameters():
                parameter.requires_grad = True


def metrics(target, prediction, labels):
    return {
        "accuracy": accuracy_score(target, prediction),
        "balanced_accuracy": balanced_accuracy_score(target, prediction),
        "macro_f1": f1_score(target, prediction, average="macro", zero_division=0),
        "confusion": confusion_matrix(target, prediction, labels=range(len(labels))).tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, device, labels):
    model.eval(); truth, predicted = [], []
    for x, y in loader:
        prediction = model(x.to(device)).argmax(-1).cpu()
        truth.extend(y.tolist()); predicted.extend(prediction.tolist())
    return metrics(truth, predicted, labels)


def train_one(protocol, arrays, kind, model_args, source, args, labels, device):
    train_x, train_y, val_x, val_y, test_x, test_y = arrays
    model = make_model(kind, model_args, len(labels)).to(device)
    if protocol != "scratch":
        initialize_from_checkpoint(model, source, kind)
    set_trainable(model, protocol)
    counts = np.bincount(train_y, minlength=len(labels))
    weights = counts.sum() / np.maximum(counts, 1) / len(labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32,
                                                         device=device))
    lr = args.head_lr if protocol == "linear_probe" else args.lr
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x),
                              torch.from_numpy(train_y)), batch_size=args.batch_size,
                              shuffle=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
                            batch_size=args.batch_size * 2)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)),
                             batch_size=args.batch_size * 2)
    best, stale, best_state = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train(); total, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            total += loss.item() * len(x); count += len(x)
        validation = evaluate(model, val_loader, device, labels)
        print(f"{protocol} {epoch}: loss={total / count:.4f} "
              f"val_f1={validation['macro_f1']:.3f}", flush=True)
        if validation["macro_f1"] > best + 1e-4:
            best, stale = validation["macro_f1"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, evaluate(model, val_loader, device, labels), evaluate(model, test_loader, device, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Extracted MAHG-EMG directory containing EMGData1.csv ...")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Neuromuscular-future or causal-GRU emg_imu_best.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mahg_emg_transfer"))
    parser.add_argument("--protocols", nargs="+", choices=["scratch", "linear_probe", "finetune"],
                        default=["scratch", "linear_probe", "finetune"])
    parser.add_argument("--files-per-subject", type=int, default=10)
    parser.add_argument("--raw-rate-hz", type=float, default=1259.0)
    parser.add_argument("--output-rate-hz", type=float, default=100.0)
    parser.add_argument("--window-ms", type=float, default=300.0)
    parser.add_argument("--stride-ms", type=float, default=50.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    files = discover(args.root)
    if not files:
        parser.error(f"no EMGData<number>.csv files found below {args.root}")
    if len(files) != 100:
        print(f"WARNING: the published MAHG-EMG dataset has 100 CSV files; found {len(files)}",
              flush=True)
    subjects = sorted({subject_for(path, args.files_per_subject) for path in files})
    if len(subjects) < 3:
        parser.error("at least three subjects are required for train/validation/test transfer")
    split = {"train": subjects[:-2], "validation": [subjects[-2]], "test": [subjects[-1]]}
    print(f"subject-held-out split: {split}", flush=True)
    rows = {name: [] for name in split}
    window_steps = round(args.window_ms * args.output_rate_hz / 1000)
    stride_steps = round(args.stride_ms * args.output_rate_hz / 1000)
    for index, path in enumerate(files, 1):
        subject = subject_for(path, args.files_per_subject)
        subset = next(name for name, members in split.items() if subject in members)
        features, flags, labels, valid = preprocess(
            path, args.raw_rate_hz, args.output_rate_hz)
        x, y = make_windows(features, flags, labels, valid, window_steps, stride_steps)
        rows[subset].append((x, y))
        print(f"[{index}/{len(files)}] subject={subject} {path.name}: {len(x)} windows", flush=True)
    label_names = sorted({str(label) for group in rows.values() for _, y in group for label in y})
    if len(label_names) != 5:
        parser.error(f"expected five GESTURE labels, found {label_names}")
    label_index = {name: index for index, name in enumerate(label_names)}
    packed = {}
    for subset, group in rows.items():
        packed[subset] = (np.concatenate([x for x, _ in group]),
                          np.asarray([label_index[str(v)] for _, y in group for v in y], dtype="int64"))
    # Domain normalization is estimated from training subjects only. Validity flags stay binary.
    mean = packed["train"][0][..., :8].mean((0, 1))
    std = packed["train"][0][..., :8].std((0, 1)).clip(1e-6)
    for subset in packed:
        packed[subset][0][..., :8] = (packed[subset][0][..., :8] - mean) / std
    arrays = (*packed["train"], *packed["validation"], *packed["test"])
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    try:
        kind, model_args = source_spec(source)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {"protocol": {"dataset": "MAHG-EMG", "root": str(args.root),
               "source_checkpoint": str(args.checkpoint), "labels": label_names,
               "source_encoder": kind,
               "subject_split": split, "files_per_subject": args.files_per_subject,
               "causal_window_ms": args.window_ms, "normalization": "training-subjects-only"}}
    for protocol in args.protocols:
        model, validation, test = train_one(
            protocol, arrays, kind, model_args, source, args, label_names, device)
        results[protocol] = {"validation": validation, "test": test}
        torch.save({"format": "mahg_emg_transfer_v1", "protocol": protocol,
                    "state_dict": model.state_dict(), "model_args": model_args,
                    "encoder_kind": kind, "source_format": source.get("format"),
                    "labels": label_names, "normalization": {"mean": mean, "std": std},
                    "preprocessing": {"raw_rate_hz": args.raw_rate_hz,
                                      "output_rate_hz": args.output_rate_hz,
                                      "window_ms": args.window_ms}},
                   args.output_dir / f"{protocol}_best.pt")
        print(protocol, "TEST", test, flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
