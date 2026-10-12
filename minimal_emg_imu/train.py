#!/usr/bin/env python3
"""Train the final causal EMG+IMU model with its four retained objectives."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.data.click_target import add_click_target
from emg_touch.data.gripper_state import add_gripper_state
from emg_touch.data.reach_grasp import preprocess
from emg_touch.models.state_attention_loss import future_position_losses
from minimal_emg_imu.model import MinimalEMGIMUModel


def masked_mean(value, mask):
    if mask.shape != value.shape:
        mask = mask.expand_as(value)
    return value[mask].mean() if mask.any() else value.sum() * 0


def normalization(trials):
    result = {}
    for key, mask_key in (("emg", "emg_valid"), ("imu", "imu_valid"),
                          ("position", "pose_valid")):
        values = np.concatenate([trial[key] for trial in trials])
        mask = np.concatenate([trial[mask_key] for trial in trials])
        if mask.ndim == 1:
            mask = np.broadcast_to(mask[:, None], values.shape)
        mean, std = [], []
        for channel in range(values.shape[1]):
            valid = values[mask[:, channel], channel]
            if not len(valid):
                raise ValueError(f"no valid training values for {key} channel {channel}")
            mean.append(float(valid.mean()))
            floor = .01 if key == "position" else 1e-6
            std.append(float(max(valid.std(), floor)))
        result[key] = {"mean": mean, "std": std}
    return result


def batches(trials, stats, batch_size, device, shuffle=False):
    order = list(range(len(trials)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), batch_size):
        selected = [trials[index] for index in order[start:start + batch_size]]
        length = max(len(trial["time"]) for trial in selected)
        batch = {"trials": selected}

        for key in ("emg", "imu"):
            channels = len(stats[key]["mean"])
            packed = np.zeros((len(selected), length, channels * 2), dtype="float32")
            usable = np.zeros((len(selected), length), dtype=bool)
            for row, trial in enumerate(selected):
                count = len(trial["time"])
                valid = trial[f"{key}_valid"]
                standardized = ((trial[key] - np.asarray(stats[key]["mean"]))
                                / np.asarray(stats[key]["std"]))
                packed[row, :count] = np.concatenate((
                    np.where(valid, np.clip(standardized, -20, 20), 0), valid), -1)
                usable[row, :count] = valid.mean(1) >= .75
            batch[key] = torch.from_numpy(packed).to(device)
            batch[f"{key}_usable"] = torch.from_numpy(usable).to(device)

        shape = (len(selected), length)
        labels = np.zeros(shape, dtype="int64")
        label_valid = np.zeros(shape, dtype=bool)
        pose = np.zeros((*shape, 3), dtype="float32")
        pose_mask = np.zeros((*shape, 1), dtype="float32")
        click = np.zeros((*shape, 2), dtype="float32")
        click_valid = np.zeros(shape, dtype=bool)
        canvas = np.ones((len(selected), 2), dtype="float32")
        progress = np.zeros(shape, dtype="float32")
        for row, trial in enumerate(selected):
            count = len(trial["time"])
            labels[row, :count] = trial["gripper_state"]
            label_valid[row, :count] = trial["gripper_state_valid"]
            pose[row, :count] = ((trial["position"] - np.asarray(stats["position"]["mean"]))
                                 / np.asarray(stats["position"]["std"]))
            pose_mask[row, :count, 0] = trial["pose_valid"]
            progress[row, :count] = np.linspace(0, 1, count, dtype="float32")
            if "click_target" in trial:
                click[row, :count] = trial["click_target"]
                click_valid[row, :count] = True
                canvas[row] = trial["canvas_px"]
        batch.update({
            "gripper_state": torch.from_numpy(labels).to(device),
            "gripper_state_valid": torch.from_numpy(label_valid).to(device),
            "pose": torch.from_numpy(pose).to(device),
            "pose_mask": torch.from_numpy(pose_mask).to(device),
            "click_target": torch.from_numpy(click).to(device),
            "click_valid": torch.from_numpy(click_valid).to(device),
            "canvas_px": torch.from_numpy(canvas).to(device),
            "trial_progress": torch.from_numpy(progress).to(device),
        })
        yield batch


def minimal_loss(output, batch, class_weight, args):
    usable = batch["emg_usable"] & batch["imu_usable"]
    state_valid = usable & batch["gripper_state_valid"]
    state = masked_mean(F.cross_entropy(
        output["gripper_state_logits"].transpose(1, 2), batch["gripper_state"],
        weight=class_weight, reduction="none"), state_valid)
    pose_valid = usable & batch["pose_mask"][..., 0].bool()
    position = masked_mean(F.smooth_l1_loss(
        output["position"], batch["pose"], reduction="none"), pose_valid[..., None])
    click_valid = usable & batch["click_valid"]
    scaled_delta = ((output["click"] - batch["click_target"])
                    * batch["canvas_px"][:, None, :] / 100.)
    pixel_frame = F.smooth_l1_loss(
        scaled_delta, torch.zeros_like(scaled_delta), reduction="none").mean(-1)
    late_weight = .25 + 3.75 * batch["trial_progress"].pow(4)
    weighted_valid = late_weight * click_valid.float()
    pixel = ((pixel_frame * weighted_valid).sum()
             / weighted_valid.sum().clamp_min(1.))
    future, _ = future_position_losses(
        output, batch, usable, include_consistency=False)
    return (args.state_weight * state + args.position_weight * position
            + args.pixel_weight * pixel + args.future_pose_weight * future)


@torch.no_grad()
def evaluate(model, trials, stats, args):
    model.eval()
    predicted_state, target_state = [], []
    position_error, pixel_error, pixel_progress = [], [], []
    future = {h: {"model": [], "hold": []} for h in range(10, 201, 10)}
    position_mean = torch.as_tensor(stats["position"]["mean"], device=args.device)
    position_std = torch.as_tensor(stats["position"]["std"], device=args.device)
    for batch in batches(trials, stats, args.batch_size, args.device):
        output = model(batch["emg"], batch["imu"])
        usable = batch["emg_usable"] & batch["imu_usable"]
        valid = usable & batch["gripper_state_valid"]
        predicted_state.extend(output["gripper_state_logits"].argmax(-1)[valid].cpu().tolist())
        target_state.extend(batch["gripper_state"][valid].cpu().tolist())
        pose_valid = usable & batch["pose_mask"][..., 0].bool()
        predicted_position = output["position"] * position_std + position_mean
        target_position = batch["pose"] * position_std + position_mean
        position_error.extend((100 * torch.linalg.vector_norm(
            predicted_position - target_position, dim=-1)[pose_valid]).cpu().tolist())
        valid = usable & batch["click_valid"]
        delta = ((output["click"] - batch["click_target"])
                 * batch["canvas_px"][:, None, :])
        pixel_error.extend(torch.linalg.vector_norm(delta, dim=-1)[valid].cpu().tolist())
        pixel_progress.extend(batch["trial_progress"][valid].cpu().tolist())
        for step in range(1, 21):
            if step >= usable.shape[1]:
                break
            valid = usable[:, :-step] & batch["pose_mask"][:, step:, 0].bool()
            predicted = output["future_position"][:, :-step, step - 1] * position_std + position_mean
            truth = target_position[:, step:]
            current = predicted_position[:, :-step]
            future[step * 10]["model"].extend((100 * torch.linalg.vector_norm(
                predicted - truth, dim=-1)[valid]).cpu().tolist())
            future[step * 10]["hold"].extend((100 * torch.linalg.vector_norm(
                current - truth, dim=-1)[valid]).cpu().tolist())
    report = {
        "gripper_accuracy": (float(np.mean(np.asarray(predicted_state) == target_state))
                             if target_state else None),
        "gripper_macro_f1": (float(f1_score(target_state, predicted_state, average="macro"))
                             if target_state else None),
        "confusion_open_close": (confusion_matrix(
            target_state, predicted_state, labels=[0, 1]).tolist() if target_state else None),
        "position_cm": float(np.mean(position_error)) if position_error else None,
        "click_pixel_error": float(np.mean(pixel_error)) if pixel_error else None,
    }
    progress, errors = np.asarray(pixel_progress), np.asarray(pixel_error)
    report["click_pixel_error_by_quarter"] = []
    for low, high in ((0, .25), (.25, .5), (.5, .75), (.75, 1)):
        mask = (progress >= low) & ((progress < high) if high < 1 else (progress <= high))
        report["click_pixel_error_by_quarter"].append(
            float(errors[mask].mean()) if mask.any() else None)
    report["future_pose_by_ms"] = {
        str(horizon): {
            "position_cm": float(np.mean(values["model"])) if values["model"] else None,
            "hold_current_prediction_cm": float(np.mean(values["hold"])) if values["hold"] else None,
            "valid_position_frames": len(values["model"]),
        } for horizon, values in future.items()}
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--raw-rate-hz", type=float, default=1259.4)
    parser.add_argument("--state-weight", type=float, default=1.)
    parser.add_argument("--position-weight", type=float, default=1.)
    parser.add_argument("--pixel-weight", type=float, default=.35)
    parser.add_argument("--future-pose-weight", type=float, default=.5)
    args = parser.parse_args()
    if min(args.state_weight, args.position_weight, args.pixel_weight,
           args.future_pose_weight) < 0:
        parser.error("loss weights must be nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use a new or empty output directory")
    return args


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    settings = {"raw_rate_hz": args.raw_rate_hz, "rate_hz": 100.,
                "gap_s": .02, "event_origin": "auto", "event_pulse_s": .1,
                "require_events": False}
    paths = sorted({path for root in args.root
                    for path in Path(root).rglob("trial_*.csv")})
    if not paths:
        raise ValueError("no trial_*.csv files found under the supplied roots")
    trials, rejected, hashes = [], {}, {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            rejected[str(path)] = f"duplicate of {hashes[digest]}"
            continue
        hashes[digest] = str(path)
        try:
            trial = add_gripper_state(path, preprocess(path, settings), settings["gap_s"])
            try:
                add_click_target(path, trial)
            except (ValueError, KeyError) as error:
                trial["audit"]["click_target"] = f"unavailable: {error}"
            trials.append(trial)
        except (ValueError, KeyError) as error:
            rejected[str(path)] = str(error)
    if len(trials) < 20:
        raise ValueError(f"need at least 20 valid trials, found {len(trials)}")
    np.random.default_rng(args.split_seed).shuffle(trials)
    count = max(1, round(.2 * len(trials)))
    test, validation, train = trials[:count], trials[count:2 * count], trials[2 * count:]
    stats = normalization(train)
    labels = np.concatenate([
        trial["gripper_state"][trial["gripper_state_valid"]] for trial in train])
    frequency = np.bincount(labels, minlength=2)
    class_weight = torch.as_tensor(
        len(labels) / np.maximum(2 * frequency, 1), dtype=torch.float32,
        device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps({
        "accepted": {trial["path"]: trial["audit"] for trial in trials},
        "rejected": rejected}, indent=2))
    (args.output_dir / "splits.json").write_text(json.dumps({
        "train": [trial["path"] for trial in train],
        "validation": [trial["path"] for trial in validation],
        "test": [trial["path"] for trial in test]}, indent=2))
    model = MinimalEMGIMUModel().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    checkpoint_path = args.output_dir / "emg_imu_best.pt"
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in batches(train, stats, args.batch_size, args.device, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["emg"], batch["imu"])
            loss = minimal_loss(output, batch, class_weight, args)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
        validation_report = evaluate(model, validation, stats, args)
        if validation_report["gripper_macro_f1"] is None:
            raise ValueError("validation split has no valid gripper labels")
        if validation_report["position_cm"] is None:
            raise ValueError("validation split has no valid position targets")
        pixel_score = validation_report["click_pixel_error"] or 0.
        score = (validation_report["position_cm"]
                 + 5 * (1 - validation_report["gripper_macro_f1"])
                 + .01 * pixel_score)
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "selection_score": score, "validation": validation_report})
        print(f"epoch={epoch} loss={np.mean(losses):.4f} score={score:.3f} "
              f"f1={validation_report['gripper_macro_f1']:.3f} "
              f"position={validation_report['position_cm']:.2f}cm "
              f"pixel={pixel_score:.1f}px", flush=True)
        if score < best:
            best, stale = score, 0
            torch.save({
                "format": "minimal_emg_imu_v1", "state_dict": model.state_dict(),
                "model_args": model.model_args, "normalization": stats,
                "preprocessing": settings, "classes": ["open", "close"],
                "seed": args.seed, "split_seed": args.split_seed,
                "validation": validation_report,
                "losses": {
                    "state_cross_entropy": args.state_weight,
                    "current_xyz_smooth_l1": args.position_weight,
                    "late_weighted_physical_pixel_smooth_l1": args.pixel_weight,
                    "future_xyz_10_to_200ms_smooth_l1": args.future_pose_weight,
                }}, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                break
    (args.output_dir / "emg_imu_history.json").write_text(json.dumps(history, indent=2))
    state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    test_report = evaluate(model, test, stats, args)
    results = {"protocol": {
        "architecture": "minimal-causal-emg-imu-v1", "roots": args.root,
        "training_seed": args.seed, "split_seed": args.split_seed,
        "split": "trial-disjoint 60/20/20",
        "heads": ["open/close", "current XYZ", "screen XY",
                  "future XYZ 10--200 ms"], "losses": state["losses"]},
        "emg+imu": test_report}
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Final checkpoint: {checkpoint_path}")
    print(json.dumps(test_report, indent=2))


if __name__ == "__main__":
    main()
