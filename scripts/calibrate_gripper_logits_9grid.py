#!/usr/bin/env python3
"""Calibrate frozen gripper logits from nine paired open/close grid trials."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.data.click_target import add_click_target
from emg_touch.data.reach_grasp import preprocess
from emg_touch.neuromuscular_inference import load_neuromuscular_model
from scripts import train_gripper_state_pose as training


def load_class(roots, expected, settings):
    settings = {**settings, "load_pose": False, "require_events": False}
    trials, rejected = [], {}
    for path in sorted({path for root in roots for path in Path(root).rglob("trial_*.csv")}):
        try:
            trial = preprocess(path, settings)
            add_click_target(path, trial)  # used only to identify the grid cell
            # The user explicitly supplies separate open/close roots. Those roots
            # are the calibration labels; a stale or absent CSV state column must
            # not silently discard an entire class.
            trial["gripper_state"] = np.full(len(trial["time"]), expected,
                                               dtype=np.int64)
            trial["gripper_state_valid"] = np.ones(len(trial["time"]), dtype=bool)
            trial["audit"]["calibration_label_source"] = (
                "open_root" if expected == 0 else "close_root")
            trials.append(trial)
        except (KeyError, ValueError) as error:
            rejected[str(path)] = str(error)
    return trials, rejected


def grid_key(trial, decimals=3):
    return tuple(np.round(trial["click_target"], decimals).tolist())


def pair_grids(open_trials, close_trials, expected_grids, tolerance):
    if len(open_trials) != expected_grids or len(close_trials) != expected_grids:
        raise ValueError(
            f"expected {expected_grids} trials per class, found "
            f"open={len(open_trials)}, close={len(close_trials)}")
    open_xy = np.stack([trial["click_target"] for trial in open_trials])
    close_xy = np.stack([trial["click_target"] for trial in close_trials])
    distances = np.linalg.norm(open_xy[:, None] - close_xy[None, :], axis=-1)
    open_indices, close_indices = linear_sum_assignment(distances)
    pairs = []
    for open_index, close_index in zip(open_indices, close_indices):
        distance = float(distances[open_index, close_index])
        if distance > tolerance:
            raise ValueError(
                "open/close grid coordinates do not match within "
                f"--grid-match-tolerance={tolerance}: open="
                f"{grid_key(open_trials[open_index])}, close="
                f"{grid_key(close_trials[close_index])}, distance={distance:.4f}")
        midpoint = tuple(((open_xy[open_index] + close_xy[close_index]) / 2).tolist())
        pairs.append((midpoint, open_trials[open_index], close_trials[close_index],
                      distance))
    return sorted(pairs, key=lambda pair: pair[0])


def print_loading_summary(name, trials, rejected):
    print(f"{name}: accepted {len(trials)} trial(s), rejected {len(rejected)}")
    for path, reason in list(rejected.items())[:10]:
        print(f"  rejected {path}: {reason}")
    if len(rejected) > 10:
        print(f"  ... and {len(rejected) - 10} more")


@torch.no_grad()
def extract_logits(model, trials, stats, args):
    model.eval(); result = []
    for batch in training.batches(trials, stats, args.batch_size, args.device):
        output = model(batch["emg"], batch["imu"])
        valid = (batch["emg_usable"] & batch["imu_usable"]
                 & batch["gripper_state_valid"])
        for row, trial in enumerate(batch["trials"]):
            result.append({"path": trial["path"],
                "logits": output["gripper_state_logits"][row, valid[row]].detach(),
                "labels": batch["gripper_state"][row, valid[row]].detach()})
    return result


def fit(records, device, maximum_iterations, regularization):
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    close_bias = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature, close_bias], lr=.5, max_iter=maximum_iterations,
        line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(.05, 20.)
        bias = torch.stack((-close_bias / 2, close_bias / 2))
        # Each recording contributes equally regardless of duration.
        losses = [F.cross_entropy(record["logits"] / temperature + bias,
                                  record["labels"])
                  for record in records]
        loss = torch.stack(losses).mean()
        loss = loss + regularization * (log_temperature.square() + close_bias.square())
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach()), float(close_bias.detach())


def predictions(records, log_temperature=0., close_bias=0.):
    truth, predicted, trial_truth, trial_predicted = [], [], [], []
    temperature = float(np.clip(np.exp(log_temperature), .05, 20.))
    bias = torch.tensor([-close_bias / 2, close_bias / 2],
                        device=records[0]["logits"].device)
    for record in records:
        logits = record["logits"] / temperature + bias
        labels = record["labels"]
        values = logits.argmax(-1)
        truth.extend(labels.cpu().tolist()); predicted.extend(values.cpu().tolist())
        trial_truth.append(int(torch.mode(labels).values))
        trial_predicted.append(int(logits.mean(0).argmax()))
    return truth, predicted, trial_truth, trial_predicted


def metrics(values):
    truth, predicted, trial_truth, trial_predicted = values
    return {"frame_macro_f1": float(f1_score(truth, predicted, average="macro")),
            "trial_accuracy": float(np.mean(np.equal(trial_truth, trial_predicted))),
            "trial_correct": int(np.equal(trial_truth, trial_predicted).sum()),
            "trial_count": len(trial_truth)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--open-root", nargs="+", required=True)
    parser.add_argument("--close-root", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-count", type=int, default=9)
    parser.add_argument("--grid-match-tolerance", type=float, default=.08,
                        help="maximum normalized distance between paired click targets")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--maximum-iterations", type=int, default=100)
    parser.add_argument("--regularization", type=float, default=1e-3)
    args = parser.parse_args()
    if args.grid_match_tolerance <= 0:
        parser.error("--grid-match-tolerance must be positive")
    if args.output.exists():
        parser.error("--output already exists")
    base, state = load_neuromuscular_model(args.checkpoint, args.device)
    if state.get("format") != "gripper_neuromuscular_future_v1":
        parser.error("--checkpoint must be the uncalibrated neuromuscular model")
    opened, rejected_open = load_class(args.open_root, 0, state["preprocessing"])
    closed, rejected_close = load_class(args.close_root, 1, state["preprocessing"])
    print_loading_summary("open", opened, rejected_open)
    print_loading_summary("close", closed, rejected_close)
    pairs = pair_grids(opened, closed, args.grid_count, args.grid_match_tolerance)
    flat = [trial for _, open_trial, close_trial, _ in pairs
            for trial in (open_trial, close_trial)]
    extracted = extract_logits(base, flat, state["normalization"], args)
    by_path = {record["path"]: record for record in extracted}
    records_by_grid = [(key, [by_path[opened["path"]], by_path[closed["path"]]])
                       for key, opened, closed, _ in pairs]

    fold_outputs, raw_outputs, folds = [], [], []
    for fold, (held_key, held_records) in enumerate(records_by_grid):
        train_records = [record for index, (_, pair) in enumerate(records_by_grid)
                         if index != fold for record in pair]
        log_temperature, close_bias = fit(
            train_records, args.device, args.maximum_iterations, args.regularization)
        raw_outputs.append(predictions(held_records))
        fold_outputs.append(predictions(held_records, log_temperature, close_bias))
        folds.append({"held_grid": held_key, "log_temperature": log_temperature,
                      "temperature": float(np.exp(log_temperature)),
                      "close_bias": close_bias,
                      "baseline": metrics(raw_outputs[-1]),
                      "calibrated": metrics(fold_outputs[-1])})

    def combine(outputs):
        return tuple(sum((list(value[index]) for value in outputs), []) for index in range(4))

    cross_validation = {"baseline": metrics(combine(raw_outputs)),
                        "calibrated": metrics(combine(fold_outputs)), "folds": folds}
    final_log_temperature, final_close_bias = fit(
        extracted, args.device, args.maximum_iterations, args.regularization)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "gripper_logit_calibration_v1",
        "base_format": state["format"], "base_state_dict": state["state_dict"],
        "model_args": state["model_args"], "normalization": state["normalization"],
        "preprocessing": state["preprocessing"],
        "log_temperature": final_log_temperature,
        "temperature": float(np.exp(final_log_temperature)),
        "close_bias": final_close_bias,
        "calibration_scope": ["gripper_logits"],
        "base_classifier_frozen": True, "pixel_pose_future_frozen": True,
        "vive_required": False,
        "calibration_label_source": "open_root/close_root",
        "grid_match_tolerance": args.grid_match_tolerance,
        "grid_pairs": [key for key, _, _, _ in pairs],
        "grid_match_distances": [distance for _, _, _, distance in pairs],
        "cross_validation": cross_validation,
        "rejected": {"open": rejected_open, "close": rejected_close}}, args.output)
    print("LEAVE-ONE-GRID-OUT", json.dumps(cross_validation, indent=2))
    print("FINAL ADAPTER", json.dumps({"temperature": float(np.exp(final_log_temperature)),
                                      "close_bias": final_close_bias}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
