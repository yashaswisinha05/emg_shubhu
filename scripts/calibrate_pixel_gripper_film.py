#!/usr/bin/env python3
"""Fit a tiny participant adapter for pixel and gripper outputs only."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.data.click_target import add_click_target
from emg_touch.data.gripper_state import add_gripper_state
from emg_touch.data.reach_grasp import preprocess
from emg_touch.models.pixel_gripper_film import PixelGripperFiLM
from emg_touch.neuromuscular_inference import load_neuromuscular_model
from scripts import train_gripper_state_pose as training


def load_trials(roots, settings):
    trials, rejected = [], {}
    settings = {**settings, "load_pose": False, "require_events": False}
    for path in sorted({p for root in roots for p in Path(root).rglob("trial_*.csv")}):
        try:
            trial = add_gripper_state(path, preprocess(path, settings), settings["gap_s"])
            add_click_target(path, trial)
            trials.append(trial)
        except (KeyError, ValueError) as error:
            rejected[str(path)] = str(error)
    return trials, rejected


def calibration_loss(output, batch, class_weight, pixel_weight, regularization, model):
    wearable = batch["emg_usable"] & batch["imu_usable"]
    state_valid = wearable & batch["gripper_state_valid"]
    state = F.cross_entropy(output["gripper_state_logits"].transpose(1, 2),
                            batch["gripper_state"], weight=class_weight,
                            reduction="none")
    state = training.masked_mean(state, state_valid)
    click_valid = wearable & batch["click_valid"]
    error = (output["click"] - batch["click_target"]) * (
        batch["canvas_px"][:, None] / 100.)
    pixel = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none").mean(-1)
    progress_weight = .1 + 2.9 * batch["trial_progress"].square()
    weights = click_valid.float() * progress_weight
    pixel = (pixel * weights).sum() / weights.sum().clamp_min(1)
    return state + pixel_weight * pixel + regularization * model.regularization()


@torch.no_grad()
def evaluate(model, trials, stats, args):
    model.eval()
    truth, predicted, errors, late_errors = [], [], [], []
    for batch in training.batches(trials, stats, args.batch_size, args.device):
        output = model(batch["emg"], batch["imu"])
        wearable = batch["emg_usable"] & batch["imu_usable"]
        valid = wearable & batch["gripper_state_valid"]
        truth.extend(batch["gripper_state"][valid].cpu().tolist())
        predicted.extend(output["gripper_state_logits"].argmax(-1)[valid].cpu().tolist())
        click_valid = wearable & batch["click_valid"]
        distance = torch.linalg.vector_norm(
            (output["click"] - batch["click_target"]) * batch["canvas_px"][:, None],
            dim=-1)
        errors.extend(distance[click_valid].cpu().tolist())
        late = click_valid & (batch["trial_progress"] >= .75)
        late_errors.extend(distance[late].cpu().tolist())
    return {
        "gripper_macro_f1": float(f1_score(truth, predicted, average="macro")),
        "pixel_error": float(np.mean(errors)),
        "late_pixel_error": float(np.mean(late_errors)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--film-groups", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--pixel-weight", type=float, default=.35)
    parser.add_argument("--identity-regularization", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; choose a new calibration checkpoint")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    base_model, state = load_neuromuscular_model(args.checkpoint, args.device)
    if state.get("format") != "gripper_neuromuscular_future_v1":
        parser.error("--checkpoint must be the uncalibrated neuromuscular checkpoint")
    trials, rejected = load_trials(args.root, state["preprocessing"])
    if len(trials) < 15:
        raise ValueError(f"need at least 15 valid calibration trials, found {len(trials)}")
    np.random.default_rng(args.seed).shuffle(trials)
    holdout = max(2, round(len(trials) / 6))
    test, validation, train = trials[:holdout], trials[holdout:2 * holdout], trials[2 * holdout:]
    labels = np.concatenate([trial["gripper_state"][trial["gripper_state_valid"]]
                             for trial in train])
    frequency = np.bincount(labels, minlength=2)
    class_weight = torch.as_tensor(len(labels) / np.maximum(2 * frequency, 1),
                                   dtype=torch.float32, device=args.device)
    model = PixelGripperFiLM(base_model, args.film_groups).to(args.device)
    optimizer = torch.optim.AdamW(model.calibration_parameters(),
                                  lr=args.learning_rate, weight_decay=0.)
    baseline = evaluate(model, test, state["normalization"], args)
    best_score, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in training.batches(train, state["normalization"],
                                      args.batch_size, args.device, True):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["emg"], batch["imu"])
            value = calibration_loss(output, batch, class_weight, args.pixel_weight,
                                     args.identity_regularization, model)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.calibration_parameters(), 1.)
            optimizer.step(); losses.append(value.item())
        report = evaluate(model, validation, state["normalization"], args)
        score = report["late_pixel_error"] / 100 + 5 * (1 - report["gripper_macro_f1"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "score": score, "validation": report})
        print(f"epoch={epoch} loss={np.mean(losses):.4f} "
              f"val_late_pixel={report['late_pixel_error']:.1f}px "
              f"val_gripper_f1={report['gripper_macro_f1']:.4f}", flush=True)
        if score < best_score:
            best_score, stale = score, 0
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.calibration_state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_calibration_state_dict(best_state)
    calibrated = evaluate(model, test, state["normalization"], args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "gripper_pixel_film_calibration_v1",
        "base_format": state["format"],
        "base_state_dict": state["state_dict"],
        "model_args": state["model_args"],
        "normalization": state["normalization"],
        "preprocessing": state["preprocessing"],
        "film_groups": args.film_groups,
        "calibration_state_dict": best_state,
        "calibration_scope": ["pixel", "gripper"],
        "calibration_inputs": ["emg", "imu", "gripper_state", "pixel_target"],
        "vive_required": False,
        "pose_and_future_frozen": True,
        "roots": args.root,
        "splits": {"train": [t["path"] for t in train],
                   "validation": [t["path"] for t in validation],
                   "test": [t["path"] for t in test]},
        "baseline_test": baseline,
        "calibrated_test": calibrated,
        "history": history,
        "rejected": rejected,
    }, args.output)
    print("BASELINE TEST", json.dumps(baseline, indent=2))
    print("CALIBRATED TEST", json.dumps(calibrated, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
