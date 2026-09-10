#!/usr/bin/env python3
"""Calibrate only the gripper branch using 20 open + 20 close trials."""
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

from emg_touch.data.gripper_state import add_gripper_state
from emg_touch.data.reach_grasp import preprocess
from emg_touch.models.hard_contrastive import hard_trial_contrastive_loss
from emg_touch.models.pixel_gripper_film import GripperHardContrastiveFiLM
from emg_touch.neuromuscular_inference import load_neuromuscular_model
from scripts import train_gripper_state_pose as training


def load_class(roots, expected, settings):
    settings = {**settings, "load_pose": False, "require_events": False}
    accepted, rejected = [], {}
    paths = sorted({path for root in roots for path in Path(root).rglob("trial_*.csv")})
    for path in paths:
        try:
            trial = add_gripper_state(path, preprocess(path, settings), settings["gap_s"])
            labels = trial["gripper_state"][trial["gripper_state_valid"]]
            if not len(labels):
                raise ValueError("no valid gripper labels")
            dominant = int(np.bincount(labels, minlength=2).argmax())
            if dominant != expected:
                raise ValueError(f"dominant state is {['open', 'close'][dominant]}")
            accepted.append(trial)
        except (KeyError, ValueError) as error:
            rejected[str(path)] = str(error)
    return accepted, rejected


def stratified_split(open_trials, close_trials, count, seed):
    rng = np.random.default_rng(seed)
    rng.shuffle(open_trials); rng.shuffle(close_trials)
    if min(len(open_trials), len(close_trials)) < count:
        raise ValueError(f"need {count} valid trials per class; found "
                         f"open={len(open_trials)}, close={len(close_trials)}")
    open_trials, close_trials = open_trials[:count], close_trials[:count]
    test_count = max(2, round(count * .15))
    validation_count = test_count
    split = {}
    for name, start, stop in (
            ("test", 0, test_count),
            ("validation", test_count, test_count + validation_count),
            ("train", test_count + validation_count, count)):
        split[name] = open_trials[start:stop] + close_trials[start:stop]
    return split


def balanced_order(trials, seed):
    rng = np.random.default_rng(seed)
    groups = {0: [], 1: []}
    for trial in trials:
        labels = trial["gripper_state"][trial["gripper_state_valid"]]
        groups[int(np.bincount(labels, minlength=2).argmax())].append(trial)
    rng.shuffle(groups[0]); rng.shuffle(groups[1])
    return [trial for pair in zip(groups[0], groups[1]) for trial in pair]


def objective(output, batch, class_weight, contrastive_weight, margin,
              identity_weight, model):
    valid = (batch["emg_usable"] & batch["imu_usable"]
             & batch["gripper_state_valid"])
    classification = F.cross_entropy(
        output["gripper_state_logits"].transpose(1, 2), batch["gripper_state"],
        weight=class_weight, reduction="none")
    classification = training.masked_mean(classification, valid)
    contrastive = hard_trial_contrastive_loss(
        output["gripper_calibration_features"], batch["gripper_state"],
        valid, margin)
    return (classification + contrastive_weight * contrastive
            + identity_weight * model.regularization()), contrastive


@torch.no_grad()
def evaluate(model, trials, stats, args):
    model.eval(); truth, predicted, probabilities = [], [], []
    for batch in training.batches(trials, stats, args.batch_size, args.device):
        output = model(batch["emg"], batch["imu"])
        valid = (batch["emg_usable"] & batch["imu_usable"]
                 & batch["gripper_state_valid"])
        truth.extend(batch["gripper_state"][valid].cpu().tolist())
        predicted.extend(output["gripper_state_logits"].argmax(-1)[valid].cpu().tolist())
        probabilities.extend(output["gripper_state_logits"].softmax(-1)[..., 1][valid]
                             .cpu().tolist())
    return {"macro_f1": float(f1_score(truth, predicted, average="macro")),
            "open_accuracy": float(np.mean(np.asarray(predicted)[np.asarray(truth) == 0] == 0)),
            "close_accuracy": float(np.mean(np.asarray(predicted)[np.asarray(truth) == 1] == 1)),
            "mean_close_probability": float(np.mean(probabilities))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--open-root", nargs="+", required=True)
    parser.add_argument("--close-root", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials-per-class", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--film-groups", type=int, default=16)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--contrastive-weight", type=float, default=.5)
    parser.add_argument("--contrastive-margin", type=float, default=.35)
    parser.add_argument("--identity-regularization", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists")
    if args.batch_size < 4 or args.batch_size % 2:
        parser.error("--batch-size must be an even number of at least four")
    if args.trials_per_class < 10:
        parser.error("use at least 10 trials per class")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    base_model, state = load_neuromuscular_model(args.checkpoint, args.device)
    if state.get("format") != "gripper_neuromuscular_future_v1":
        parser.error("--checkpoint must be the uncalibrated neuromuscular model")
    open_trials, open_rejected = load_class(
        args.open_root, 0, state["preprocessing"])
    close_trials, close_rejected = load_class(
        args.close_root, 1, state["preprocessing"])
    split = stratified_split(open_trials, close_trials,
                             args.trials_per_class, args.seed)
    print("balanced trials:", {name: len(value) for name, value in split.items()})
    model = GripperHardContrastiveFiLM(
        base_model, args.film_groups, args.adapter_rank).to(args.device)
    class_weight = torch.ones(2, device=args.device)
    optimizer = torch.optim.AdamW(model.gripper_calibration_parameters(),
                                  lr=args.learning_rate, weight_decay=0.)
    baseline = evaluate(model, split["test"], state["normalization"], args)
    best_score, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses, contrasts = [], []
        order = balanced_order(split["train"], args.seed + epoch)
        for batch in training.batches(order, state["normalization"],
                                      args.batch_size, args.device, False):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["emg"], batch["imu"])
            value, contrast = objective(
                output, batch, class_weight, args.contrastive_weight,
                args.contrastive_margin, args.identity_regularization, model)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.gripper_calibration_parameters(), 1.)
            optimizer.step(); losses.append(value.item()); contrasts.append(contrast.item())
        report = evaluate(model, split["validation"], state["normalization"], args)
        score = 1 - report["macro_f1"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "hard_contrastive": float(np.mean(contrasts)),
                        "validation": report})
        print(f"epoch={epoch} loss={np.mean(losses):.4f} "
              f"hard={np.mean(contrasts):.4f} val_f1={report['macro_f1']:.4f}")
        if score < best_score:
            best_score, stale = score, 0
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.calibration_state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_calibration_state_dict(best_state)
    calibrated = evaluate(model, split["test"], state["normalization"], args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "gripper_hard_contrastive_calibration_v1",
        "base_format": state["format"], "base_state_dict": state["state_dict"],
        "model_args": state["model_args"], "normalization": state["normalization"],
        "preprocessing": state["preprocessing"], "film_groups": args.film_groups,
        "adapter_rank": args.adapter_rank, "calibration_state_dict": best_state,
        "calibration_scope": ["gripper"], "pixel_pose_future_frozen": True,
        "vive_required": False, "trials_per_class": args.trials_per_class,
        "splits": {name: [trial["path"] for trial in trials]
                   for name, trials in split.items()},
        "baseline_test": baseline, "calibrated_test": calibrated,
        "history": history,
        "rejected": {"open": open_rejected, "close": close_rejected}}, args.output)
    print("BASELINE TEST", json.dumps(baseline, indent=2))
    print("CALIBRATED TEST", json.dumps(calibrated, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
