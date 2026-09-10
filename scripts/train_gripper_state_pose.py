#!/usr/bin/env python3
"""Train causal EMG+IMU pose regression and dense open/close classification."""
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
from emg_touch.models.reach_grasp_gripper_state import GripperStatePoseModel
from emg_touch.physics.rotation_6d import orientation_errors_numpy
from scripts import train_reach_grasp as base


def batches(trials, stats, batch_size, device, shuffle=False):
    for batch in base.batches(trials, stats, batch_size, device, shuffle):
        shape = batch["emg"].shape[:2]
        labels = np.zeros(shape, dtype="int64")
        valid = np.zeros(shape, dtype=bool)
        click = np.zeros((*shape, 2), dtype="float32")
        click_valid = np.zeros(shape, dtype=bool)
        canvas = np.ones((shape[0], 2), dtype="float32")
        progress = np.zeros(shape, dtype="float32")
        for row, trial in enumerate(batch["trials"]):
            n = len(trial["time"])
            labels[row, :n] = trial["gripper_state"]
            valid[row, :n] = trial["gripper_state_valid"]
            progress[row, :n] = np.linspace(0., 1., n, dtype="float32")
            if "click_target" in trial:
                click[row, :n] = trial["click_target"]
                click_valid[row, :n] = True
                canvas[row] = trial["canvas_px"]
        batch["gripper_state"] = torch.from_numpy(labels).to(device)
        batch["gripper_state_valid"] = torch.from_numpy(valid).to(device)
        batch["click_target"] = torch.from_numpy(click).to(device)
        batch["click_valid"] = torch.from_numpy(click_valid).to(device)
        batch["canvas_px"] = torch.from_numpy(canvas).to(device)
        batch["trial_progress"] = torch.from_numpy(progress).to(device)
        yield batch


def span_mask(shape, device, probability=.25, span=10):
    blocks = (shape[1] + span - 1) // span
    return (torch.rand(shape[0], blocks, device=device) < probability).repeat_interleave(
        span, 1)[:, :shape[1]]


def masked_mean(values, mask):
    if mask.shape != values.shape:
        mask = mask.expand_as(values)
    return values[mask].mean() if mask.any() else values.sum() * 0


def loss(model, output, batch, class_weight, args):
    wearable = batch["emg_usable"] & batch["imu_usable"]
    valid = wearable & batch["gripper_state_valid"]
    labels = batch["gripper_state"]
    state = masked_mean(F.cross_entropy(output["gripper_state_logits"].transpose(1, 2),
                                        labels, weight=class_weight, reduction="none"), valid)
    react = masked_mean(F.cross_entropy(output["react_gripper_state_logits"].transpose(1, 2),
                                        labels, weight=class_weight, reduction="none"), valid)
    hidden = span_mask(labels.shape, labels.device)
    actions = torch.where(hidden & valid, torch.full_like(labels, 2), labels)
    actions = torch.where(valid, actions, torch.full_like(labels, 2))
    emg_hidden = span_mask(labels.shape, labels.device)
    auxiliary = model.react(batch["emg"], actions=actions, hidden=emg_hidden)
    reconstruct_mask = emg_hidden[..., None] & batch["emg"][..., 8:].bool()
    reconstruction = masked_mean(
        (auxiliary["reconstruction"] - batch["emg"][..., :8]).square(), reconstruct_mask)
    hidden_valid = hidden & valid
    masked_state = masked_mean(F.cross_entropy(auxiliary["holding_logits"].transpose(1, 2),
                               labels, weight=class_weight, reduction="none"), hidden_valid)
    pair = valid[:, 1:] & valid[:, :-1] & (labels[:, 1:] == labels[:, :-1])
    probability = output["gripper_state_logits"].softmax(-1)[..., 1]
    stability = masked_mean((probability[:, 1:] - probability[:, :-1]).square(), pair)
    pose_valid = wearable & batch["pose_mask"].squeeze(-1).bool()
    position = masked_mean(F.smooth_l1_loss(output["position"], batch["pose"],
                                            reduction="none"), pose_valid[..., None])
    orientation_valid = wearable & batch["orientation_mask"]
    orientation = masked_mean(F.smooth_l1_loss(output["orientation_6d"], batch["orientation"],
                                               reduction="none"), orientation_valid[..., None])
    total = (state + args.react_weight * react + args.masked_weight * (masked_state + reconstruction)
             + args.stability_weight * stability + args.position_weight * position
             + args.orientation_weight * orientation)
    if output["click"] is not None:
        click_valid = wearable & batch["click_valid"]
        click = masked_mean(F.smooth_l1_loss(output["click"], batch["click_target"],
                                             reduction="none"), click_valid[..., None])
        total = total + args.pixel_weight * click
    return total


@torch.no_grad()
def evaluate(model, trials, stats, args, zero_emg=False, zero_imu=False):
    model.eval()
    predicted, target, position_error, angle_error = [], [], [], []
    pixel_error, pixel_progress = [], []
    for batch in batches(trials, stats, args.batch_size, args.device):
        emg = torch.zeros_like(batch["emg"]) if zero_emg else batch["emg"]
        imu = torch.zeros_like(batch["imu"]) if zero_imu else batch["imu"]
        output = model(emg, imu)
        wearable = batch["emg_usable"] & batch["imu_usable"]
        state_valid = wearable & batch["gripper_state_valid"]
        predicted.extend(output["gripper_state_logits"].argmax(-1)[state_valid].cpu().tolist())
        target.extend(batch["gripper_state"][state_valid].cpu().tolist())
        pose_valid = wearable & batch["pose_mask"].squeeze(-1).bool()
        p = output["position"] * torch.as_tensor(stats["position"]["std"], device=args.device)
        p += torch.as_tensor(stats["position"]["mean"], device=args.device)
        truth = batch["pose"] * torch.as_tensor(stats["position"]["std"], device=args.device)
        truth += torch.as_tensor(stats["position"]["mean"], device=args.device)
        position_error.extend((100 * torch.linalg.vector_norm(p - truth, dim=-1)[pose_valid]).cpu().tolist())
        orientation_valid = wearable & batch["orientation_mask"]
        if orientation_valid.any():
            geodesic, _ = orientation_errors_numpy(
                output["orientation_6d"][orientation_valid].cpu().numpy(),
                batch["orientation"][orientation_valid].cpu().numpy())
            angle_error.extend(geodesic.tolist())
        if output["click"] is not None:
            click_valid = wearable & batch["click_valid"]
            if click_valid.any():
                delta = (output["click"] - batch["click_target"]) * batch["canvas_px"][:, None, :]
                distance = torch.linalg.vector_norm(delta, dim=-1)
                pixel_error.extend(distance[click_valid].cpu().tolist())
                pixel_progress.extend(batch["trial_progress"][click_valid].cpu().tolist())
    report = {"gripper_accuracy": float(np.mean(np.equal(predicted, target))),
              "gripper_macro_f1": float(f1_score(target, predicted, average="macro")),
              "confusion_open_close": confusion_matrix(target, predicted, labels=[0, 1]).tolist(),
              "position_cm": float(np.mean(position_error)),
              "orientation_deg": float(np.mean(angle_error)) if angle_error else None}
    if pixel_error:
        error, progress = np.asarray(pixel_error), np.asarray(pixel_progress)
        report["click_pixel_error"] = float(error.mean())
        # One target per trial, so the model predicts the same coordinate at
        # every timestep: the quarters show how early in the reach it converges.
        quarters = []
        for low, high in ((0., .25), (.25, .5), (.5, .75), (.75, 1.001)):
            window = (progress >= low) & (progress < high)
            quarters.append(float(error[window].mean()) if window.any() else None)
        report["click_pixel_error_by_quarter"] = quarters
    return report


def train_one(modality, args, train, validation, stats, class_weight, settings, augmenter=None):
    """Train one dedicated model restricted to ``modality`` end to end.

    Mirrors train_reach_grasp.py's --models loop: each modality gets its own
    model instance (GripperStatePoseModel(modality=...) zeroes and one-hots
    the fusion gate for the unused branch), its own optimizer, its own
    checkpoint/history files, and the same reseeded init so the three runs
    are a fair capacity-matched comparison rather than one model probed with
    its inputs zeroed out at test time.
    """
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model_args = {"modality": modality, "width": 128, "patch": 16, "stride": 4,
                  "layers": 4, "heads": 4, "dropout": .1, "react_context": 100,
                  "predict_click": args.pixel_weight > 0}
    model = GripperStatePoseModel(**model_args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    suffix = modality.replace("+", "_")
    checkpoint = args.output_dir / f"{suffix}_best.pt"
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in batches(train, stats, args.batch_size, args.device, True):
            if augmenter is not None:
                batch = augmenter(batch, stats, modality)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["emg"], batch["imu"])
            value = loss(model, output, batch, class_weight, args)
            if not torch.isfinite(value): raise FloatingPointError("nonfinite loss")
            value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            losses.append(value.item())
        report = evaluate(model, validation, stats, args)
        score = report["position_cm"] + .1 * (report["orientation_deg"] or 0) + 5 * (1-report["gripper_macro_f1"])
        # Pixels are a much larger number than centimetres, so scale it into
        # the same range as the other terms before it drives checkpointing.
        score += .01 * report.get("click_pixel_error", 0.)
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "selection_score": score, "validation": report})
        pixels = report.get("click_pixel_error")
        print(f"modality={modality} epoch={epoch} loss={np.mean(losses):.4f} score={score:.3f} "
              f"state_f1={report['gripper_macro_f1']:.3f} pose={report['position_cm']:.2f}cm"
              + (f" click={pixels:.1f}px" if pixels is not None else ""), flush=True)
        if score < best:
            best, stale = score, 0
            torch.save({"format": "gripper_state_pose_v2", "state_dict": model.state_dict(),
                "model_args": model_args, "normalization": stats, "preprocessing": settings,
                "classes": ["open", "close"], "validation": report, "seed": args.seed,
                "augmentation": {"physiological": augmenter is not None,
                                 "strength": args.augmentation_strength}}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience: break
    (args.output_dir / f"{suffix}_history.json").write_text(json.dumps(history, indent=2))
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    return model, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", nargs="+", required=True,
                        help="One or more dataset directories, searched recursively "
                             "for trial_*.csv; pass two or more to pool datasets into "
                             "one training run (byte-identical trials across roots "
                             "are still deduplicated by content hash)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gripper_state_pose_seed42"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-rate-hz", type=float, default=1259.4)
    parser.add_argument("--event-origin", choices=["auto", "start"], default="auto")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--react-weight", type=float, default=.2)
    parser.add_argument("--masked-weight", type=float, default=.15)
    parser.add_argument("--stability-weight", type=float, default=.03)
    parser.add_argument("--position-weight", type=float, default=.2)
    parser.add_argument("--orientation-weight", type=float, default=.5)
    parser.add_argument("--pixel-weight", type=float, default=.2,
                        help="Weight on predicting the trial's click target in normalized "
                             "canvas units, reported back as pixel error. Set to 0 to skip "
                             "the head entirely rather than instantiate an untrained one")
    parser.add_argument("--physiological-augmentation", action="store_true",
                        help="Apply causal EMG/IMU gain, mounting-rotation, noise, drift and "
                             "dropout augmentation to training batches only (see "
                             "emg_touch.data.wearable_augmentation). These are exactly the "
                             "nuisance factors that differ between two donnings of the band, "
                             "so this makes a session-constant shortcut harder to exploit")
    parser.add_argument("--augmentation-strength", type=float, default=1.)
    parser.add_argument("--models", nargs="+", choices=["emg", "imu", "emg+imu"],
                        default=["emg", "imu", "emg+imu"],
                        help="Trains one dedicated model per entry (matches "
                             "train_reach_grasp.py's convention); pass a single "
                             "value to skip the ablation and train only that one")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use an empty output directory")
    if args.augmentation_strength < 0:
        parser.error("augmentation strength cannot be negative")
    if args.pixel_weight < 0:
        parser.error("pixel weight cannot be negative")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    settings = {"raw_rate_hz": args.raw_rate_hz, "rate_hz": 100., "gap_s": .02,
                "event_origin": args.event_origin, "event_pulse_s": .1,
                "require_events": False}
    trials, rejected, hashes = [], {}, {}
    paths = sorted({p for root in args.root for p in Path(root).rglob("trial_*.csv")})
    if not paths:
        raise ValueError(
            "no files matched 'trial_*.csv' (recursively) under: "
            + ", ".join(args.root)
            + " -- check the directory paths and that trial files are named trial_*.csv")
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            rejected[str(path)] = "byte-identical duplicate of " + hashes[digest]; continue
        hashes[digest] = str(path)
        try:
            trial = add_gripper_state(path, preprocess(path, settings), settings["gap_s"])
            if args.pixel_weight:
                # A trial without click columns still trains everything else;
                # the pixel term simply masks it out.
                try:
                    add_click_target(path, trial)
                except (ValueError, KeyError) as error:
                    trial["audit"]["click_target"] = f"unavailable: {error}"
            trials.append(trial)
        except (ValueError, KeyError) as error:
            rejected[str(path)] = str(error)
    if len(trials) < 20:
        # Surface WHY, before raising -- data_audit.json previously was only
        # written after this check, so a failed run gave zero diagnostic
        # information beyond "need at least 20 valid trials".
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "data_audit.json").write_text(json.dumps(
            {"accepted": {t["path"]: t["audit"] for t in trials}, "rejected": rejected}, indent=2))
        reasons = {}
        for message in rejected.values():
            reasons[message] = reasons.get(message, 0) + 1
        summary = "\n".join(f"  {count:4d}x  {reason}" for reason, count in
                            sorted(reasons.items(), key=lambda kv: -kv[1])[:10])
        raise ValueError(
            f"need at least 20 valid trials, found {len(trials)} valid out of "
            f"{len(paths)} files matched under {', '.join(args.root)}\n"
            f"rejection reasons (see {args.output_dir / 'data_audit.json'} for the full list):\n"
            f"{summary or '  (no files were rejected -- fewer than 20 trial_*.csv files exist)'}")
    np.random.default_rng(args.seed).shuffle(trials)
    count = max(1, round(.2 * len(trials)))
    test, validation, train = trials[:count], trials[count:2*count], trials[2*count:]
    stats = base.normalization(train)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps(
        {"accepted": {t["path"]: t["audit"] for t in trials}, "rejected": rejected}, indent=2))
    (args.output_dir / "splits.json").write_text(json.dumps({k: [t["path"] for t in v]
        for k, v in (("train", train), ("validation", validation), ("test", test))}, indent=2))
    labels = np.concatenate([t["gripper_state"][t["gripper_state_valid"]] for t in train])
    frequency = np.bincount(labels, minlength=2)
    class_weight = torch.as_tensor(len(labels) / np.maximum(2 * frequency, 1),
                                   dtype=torch.float32, device=args.device)
    if args.pixel_weight:
        targeted = [t for t in trials if "click_target" in t]
        distinct = {tuple(np.round(t["click_target"], 4)) for t in targeted}
        print(f"click targets: {len(targeted)}/{len(trials)} trials, "
              f"{len(distinct)} distinct locations", flush=True)
        if not targeted:
            raise ValueError("--pixel-weight is set but no trial carried a click target; "
                             "see data_audit.json for the per-trial reason")
        if len(distinct) < 2:
            raise ValueError("every trial shares one click target, so there is nothing "
                             "to predict; pass --pixel-weight 0")

    augmenter = None
    if args.physiological_augmentation:
        from emg_touch.data.wearable_augmentation import PhysiologicalWearableAugmenter
        augmenter = PhysiologicalWearableAugmenter(args.augmentation_strength)

    results = {"protocol": {"roots": args.root, "models": args.models,
                            "vive_role": "pose supervision only",
                            "gripper_state_role": "classification supervision only",
                            "physiological_augmentation": args.physiological_augmentation,
                            "augmentation_strength": args.augmentation_strength,
                            "pixel_weight": args.pixel_weight,
                            "click_target_role": "screen-target regression supervision",
                            "preprocessing": settings}}
    for modality in args.models:
        model, _ = train_one(modality, args, train, validation, stats, class_weight,
                             settings, augmenter)
        results[modality] = evaluate(model, test, stats, args)
        # Same jointly-trained model, but with one sensor stream zeroed at
        # test time -- a graceful-degradation check, distinct from (and
        # complementary to) the dedicated emg-only/imu-only models above,
        # which never had capacity allocated to the missing modality at all.
        if modality == "emg+imu":
            results["fusion_zero_emg"] = evaluate(model, test, stats, args, zero_emg=True)
            results["fusion_zero_imu"] = evaluate(model, test, stats, args, zero_imu=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print("TEST RESULTS", json.dumps(results, indent=2))


if __name__ == "__main__": main()
