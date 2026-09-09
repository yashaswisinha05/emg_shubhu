#!/usr/bin/env python3
"""Train EMG+IMU current-state and one-second trajectory/interaction intent."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.data.annotation_uncertainty import soften_events
from emg_touch.data.reach_grasp import preprocess
from emg_touch.models.reach_grasp_future_intent import ReachGraspFutureIntentModel
from emg_touch.physics.rotation_6d import orientation_errors_numpy
from scripts import train_reach_grasp as base


HORIZONS_MS = (100, 250, 500, 750, 1000)


def future_probability(time, event, horizon_s, uncertainty_s):
    """P(event occurs in [now, now+horizon]) under timestamp uncertainty."""
    sigma = max(.04, uncertainty_s / 2)
    delta = float(event) - np.asarray(time, dtype=float)
    return (ndtr((horizon_s - delta) / sigma) - ndtr(-delta / sigma)).astype("float32")


def time_distribution(time, events, centers_s, horizon_s, uncertainty_s):
    time = np.asarray(time, dtype=float)
    centers = np.asarray(centers_s, dtype=float)
    result = np.zeros((len(time), 2, len(centers) + 1), dtype="float32")
    sigma = max(.04, uncertainty_s / 2)
    for event_index, event in enumerate(events):
        delta = float(event) - time
        inside = (delta >= -uncertainty_s) & (delta <= horizon_s + uncertainty_s)
        score = np.exp(-.5 * ((delta[:, None] - centers[None]) / sigma) ** 2)
        score /= np.maximum(score.sum(1, keepdims=True), 1e-12)
        result[inside, event_index, :-1] = score[inside]
        result[~inside, event_index, -1] = 1
    return result


def future_batches(trials, stats, batch_size, device, horizons_ms, shuffle=False):
    horizons_s = np.asarray(horizons_ms, dtype=float) / 1000
    centers_s = np.r_[0., horizons_s]
    for batch in base.batches(trials, stats, batch_size, device, shuffle):
        count = len(batch["trials"])
        frames = batch["emg"].shape[1]
        steps = len(horizons_s)
        position = np.zeros((count, frames, steps, 3), dtype="float32")
        orientation = np.zeros((count, frames, steps, 6), dtype="float32")
        pose_mask = np.zeros((count, frames, steps), dtype=bool)
        orientation_mask = np.zeros((count, frames, steps), dtype=bool)
        event = np.zeros((count, frames, 2), dtype="float32")
        event_time = np.zeros((count, frames, 2, len(centers_s) + 1), dtype="float32")
        for row, trial in enumerate(batch["trials"]):
            n = len(trial["time"])
            uncertainty = float(trial.get("annotation_uncertainty_s", .5))
            for event_index in range(2):
                event[row, :n, event_index] = future_probability(
                    trial["time"], trial["events"][event_index],
                    horizons_s[-1], uncertainty)
            event_time[row, :n] = time_distribution(
                trial["time"], trial["events"], centers_s,
                horizons_s[-1], uncertainty)
            for step, horizon in enumerate(horizons_s):
                target_time = trial["time"] + horizon
                index = np.searchsorted(trial["time"], target_time, side="left")
                available = index < n
                clipped = np.minimum(index, n - 1)
                position[row, :n, step] = (
                    trial["position"][clipped] - stats["position"]["mean"]) / stats["position"]["std"]
                orientation[row, :n, step] = trial["orientation"][clipped]
                pose_mask[row, :n, step] = available & trial["pose_valid"][clipped]
                orientation_mask[row, :n, step] = (
                    available & trial["orientation_valid"][clipped])
        batch.update({
            "future_position": torch.from_numpy(position).to(device),
            "future_orientation": torch.from_numpy(orientation).to(device),
            "future_pose_mask": torch.from_numpy(pose_mask).to(device),
            "future_orientation_mask": torch.from_numpy(orientation_mask).to(device),
            "future_event": torch.from_numpy(event).to(device),
            "intent_time_target": torch.from_numpy(event_time).to(device)})
        yield batch


def masked_smooth_l1(predicted, target, mask):
    if not mask.any():
        return predicted.sum() * 0
    return F.smooth_l1_loss(predicted[mask], target[mask])


def future_loss(output, batch, usable, args):
    future_mask = batch["future_pose_mask"] & usable.unsqueeze(-1)
    orientation_mask = batch["future_orientation_mask"] & usable.unsqueeze(-1)
    position = masked_smooth_l1(
        output["future_position"], batch["future_position"], future_mask)
    orientation = masked_smooth_l1(
        output["future_orientation_6d"], batch["future_orientation"],
        orientation_mask)
    endpoint = masked_smooth_l1(
        output["future_endpoint"], batch["future_position"][..., -1, :],
        future_mask[..., -1])
    pair = future_mask[..., 1:] & future_mask[..., :-1]
    increments = masked_smooth_l1(
        output["future_position"][..., 1:, :] - output["future_position"][..., :-1, :],
        batch["future_position"][..., 1:, :] - batch["future_position"][..., :-1, :],
        pair)
    event_bce = F.binary_cross_entropy_with_logits(
        output["future_event_logits"], batch["future_event"], reduction="none")
    event_weight = 1 + 3 * batch["future_event"]
    event_mask = usable.unsqueeze(-1)
    event = (event_bce * event_weight * event_mask).sum() / (
        event_weight * event_mask).sum().clamp_min(1)
    target = batch["intent_time_target"]
    time_ce = -(target * output["intent_time_logits"].log_softmax(-1)).sum(-1)
    time_weight = torch.where(target[..., -1] > .5, .1, 1.)
    time_mask = usable.unsqueeze(-1)
    timing = (time_ce * time_weight * time_mask).sum() / (
        time_weight * time_mask).sum().clamp_min(1)
    total = (args.future_position_weight * position
             + args.future_orientation_weight * orientation
             + args.future_endpoint_weight * endpoint
             + args.path_shape_weight * increments
             + args.future_event_weight * event
             + args.intent_time_weight * timing)
    return total, {"future_position": position.item(),
                   "future_orientation": orientation.item(),
                   "future_endpoint": endpoint.item(), "path_shape": increments.item(),
                   "future_event": event.item(), "intent_time": timing.item()}


@torch.no_grad()
def predict(model, trials, stats, args, zero_emg=False, zero_imu=False):
    model.eval()
    predictions = []
    for batch in base.batches(trials, stats, args.batch_size, args.device):
        emg = torch.zeros_like(batch["emg"]) if zero_emg else batch["emg"]
        imu = torch.zeros_like(batch["imu"]) if zero_imu else batch["imu"]
        output = model(emg, imu)
        valid = batch["emg_usable"] & batch["imu_usable"]
        for row, trial in enumerate(batch["trials"]):
            n = len(trial["time"])
            position = output["position"][row, :n].cpu().numpy()
            position = position * stats["position"]["std"] + stats["position"]["mean"]
            future = output["future_position"][row, :n].cpu().numpy()
            future = future * np.asarray(stats["position"]["std"])[None, None] + np.asarray(
                stats["position"]["mean"])[None, None]
            predictions.append({"trial": trial, "valid": valid[row, :n].cpu().numpy(),
                "prob": output["logits"][row, :n].sigmoid().cpu().numpy(),
                "position": position,
                "orientation_6d": output["orientation_6d"][row, :n].cpu().numpy(),
                "future_position": future,
                "future_orientation_6d": output["future_orientation_6d"][row, :n].cpu().numpy(),
                "future_event_probability": output["future_event_logits"][row, :n].sigmoid().cpu().numpy(),
                "intent_time_probability": output["intent_time_logits"][row, :n].softmax(-1).cpu().numpy()})
    return predictions


def safe_binary_metric(target, probability):
    if len(np.unique(target)) < 2:
        return {"auroc": None, "average_precision": None}
    return {"auroc": float(roc_auc_score(target, probability)),
            "average_precision": float(average_precision_score(target, probability))}


def intent_metrics(predictions, horizons_ms, uncertainty_s=.5):
    position = [[] for _ in horizons_ms]
    orientation = [[] for _ in horizons_ms]
    targets, probabilities = [[], []], [[], []]
    time_errors = [[], []]
    centers = np.r_[0., np.asarray(horizons_ms) / 1000]
    for item in predictions:
        trial, valid = item["trial"], item["valid"]
        for step, horizon_ms in enumerate(horizons_ms):
            target_time = trial["time"] + horizon_ms / 1000
            index = np.searchsorted(trial["time"], target_time, side="left")
            available = index < len(trial["time"])
            clipped = np.minimum(index, len(trial["time"]) - 1)
            pm = valid & available & trial["pose_valid"][clipped]
            position[step].extend(100 * np.linalg.norm(
                item["future_position"][pm, step] - trial["position"][clipped[pm]], axis=-1))
            om = valid & available & trial["orientation_valid"][clipped]
            if om.any():
                error, _ = orientation_errors_numpy(
                    item["future_orientation_6d"][om, step], trial["orientation"][clipped[om]])
                orientation[step].extend(error)
        for event in range(2):
            delta = trial["events"][event] - trial["time"]
            target = (delta >= 0) & (delta <= horizons_ms[-1] / 1000)
            targets[event].extend(target[valid].astype(int))
            probabilities[event].extend(item["future_event_probability"][valid, event])
            inside = valid & target
            distribution = item["intent_time_probability"][:, event, :-1]
            conditional = distribution / np.maximum(distribution.sum(-1, keepdims=True), 1e-9)
            expected = (conditional * centers[None, :]).sum(-1)
            time_errors[event].extend(1000 * np.abs(expected[inside] - delta[inside]))
    result = {"future_position_cm": {str(ms): float(np.mean(value)) if value else None
                                     for ms, value in zip(horizons_ms, position)},
              "future_orientation_deg": {str(ms): float(np.mean(value)) if value else None
                                          for ms, value in zip(horizons_ms, orientation)}}
    for event, name in enumerate(["grasp", "release"]):
        result[name + "_within_1s"] = safe_binary_metric(
            np.asarray(targets[event]), np.asarray(probabilities[event]))
        result[name + "_time_mae_ms"] = (float(np.mean(time_errors[event]))
            if time_errors[event] else None)
    result["intent_mean_average_precision"] = float(np.mean([
        result[name + "_within_1s"]["average_precision"]
        for name in ["grasp", "release"]]))
    result["mean_future_position_cm"] = float(np.mean([
        value for value in result["future_position_cm"].values() if value is not None]))
    result["mean_future_orientation_deg"] = float(np.mean([
        value for value in result["future_orientation_deg"].values() if value is not None]))
    return result


def main(model_class=ReachGraspFutureIntentModel,
         checkpoint_format="reach_grasp_future_intent_v1", extension=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/reach_grasp_future_intent_seed42"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-rate-hz", type=float, default=1259.4)
    parser.add_argument("--event-origin", choices=["auto", "start"], default="auto")
    parser.add_argument("--uncertainty-ms", type=float, default=500.)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--future-position-weight", type=float, default=1.)
    parser.add_argument("--future-orientation-weight", type=float, default=.35)
    parser.add_argument("--future-endpoint-weight", type=float, default=.5)
    parser.add_argument("--path-shape-weight", type=float, default=.5)
    parser.add_argument("--future-event-weight", type=float, default=.75)
    parser.add_argument("--current-event-time-weight", type=float, default=.25)
    parser.add_argument("--intent-time-weight", type=float, default=.35)
    parser.add_argument("--split-file", type=Path,
                        help="Reuse train/validation/test trial paths from a previous run")
    parser.add_argument("--split-by", choices=["trial", "recording"], default="trial",
                        help="Recording holds out complete CSV parent directories")
    if extension is not None:
        extension.configure(parser)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use an empty output directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    settings = {"raw_rate_hz": args.raw_rate_hz, "rate_hz": 100., "gap_s": .02,
                "event_origin": args.event_origin, "event_pulse_s": .1,
                "annotation_uncertainty_s": args.uncertainty_ms / 1000}
    trials, rejected, hashes = [], {}, {}
    for path in sorted(Path(args.root).rglob("trial_*.csv")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            rejected[str(path)] = "byte-identical duplicate of " + hashes[digest]
            continue
        hashes[digest] = str(path)
        try:
            trial = preprocess(path, settings)
            trial["annotation_uncertainty_s"] = args.uncertainty_ms / 1000
            soften_events(trial, args.uncertainty_ms / 1000)
            trials.append(trial)
        except (ValueError, KeyError) as error:
            rejected[str(path)] = str(error)
    if len(trials) < 20:
        raise ValueError("need at least 20 usable trials")
    np.random.default_rng(args.seed).shuffle(trials)
    n = max(1, round(.2 * len(trials)))
    test, validation, train = trials[:n], trials[n:2*n], trials[2*n:]
    if args.split_by == "recording":
        recording_ids = sorted({str(Path(t["path"]).parent.resolve()) for t in trials})
        if len(recording_ids) < 3:
            raise ValueError("recording split requires at least three CSV parent directories")
        np.random.default_rng(args.seed).shuffle(recording_ids)
        count = max(1, round(.2 * len(recording_ids)))
        test_ids, val_ids = set(recording_ids[:count]), set(recording_ids[count:2*count])
        test = [t for t in trials if str(Path(t["path"]).parent.resolve()) in test_ids]
        validation = [t for t in trials if str(Path(t["path"]).parent.resolve()) in val_ids]
        train = [t for t in trials if str(Path(t["path"]).parent.resolve()) not in test_ids | val_ids]
    if args.split_file is not None:
        manifest = json.loads(args.split_file.read_text())
        lookup = {str(Path(t["path"]).resolve()): t for t in trials}
        groups = [[str(Path(p).resolve()) for p in manifest[k]]
                  for k in ("train", "validation", "test")]
        flat = sum(groups, [])
        if any(not g for g in groups) or len(flat) != len(set(flat)):
            raise ValueError("split file has empty or overlapping splits")
        if set(flat) != set(lookup):
            raise ValueError("split file must match all accepted trial paths exactly")
        train, validation, test = [[lookup[p] for p in g] for g in groups]
        if args.split_by == "recording":
            parents = [{str(Path(p).parent) for p in g} for g in groups]
            if any(parents[a] & parents[b] for a, b in ((0, 1), (0, 2), (1, 2))):
                raise ValueError("split file shares recordings across splits")
    stats = base.normalization(train)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps(
        {"accepted": len(trials), "rejected": rejected}, indent=2))
    (args.output_dir / "splits.json").write_text(json.dumps({
        "train": [t["path"] for t in train], "validation": [t["path"] for t in validation],
        "test": [t["path"] for t in test]}, indent=2))
    print(f"Trials: train={len(train)}, validation={len(validation)}, "
          f"test={len(test)}, rejected={len(rejected)}", flush=True)

    model_args = {"modality": "emg+imu", "width": 128, "patch": 16,
        "stride": 4, "layers": 4, "heads": 4, "dropout": .1,
        "event_time_bins": 6, "future_horizons_ms": HORIZONS_MS}
    model = model_class(**model_args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    labels = np.concatenate([np.column_stack([t["holding"], t["event_labels"]]) for t in train])
    positive = labels.sum(0)
    weights = torch.tensor(np.clip((len(labels) - positive) / np.maximum(positive, 1), 1, 50),
        dtype=torch.float32, device=args.device)
    best, stale, history = float("inf"), 0, []
    checkpoint = args.output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, components = [], []
        for batch in future_batches(
                train, stats, args.batch_size, args.device, HORIZONS_MS, True):
            usable = batch["emg_usable"] & batch["imu_usable"]
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["emg"], batch["imu"])
            interaction = F.binary_cross_entropy_with_logits(
                output["logits"], batch["labels"], pos_weight=weights, reduction="none")
            mask = usable.unsqueeze(-1) * batch["label_mask"]
            loss = (interaction * mask).sum() / mask.sum().clamp_min(1)
            current_time_target = batch["event_time_targets"]
            current_time_loss = -(current_time_target * output[
                "event_time_logits"].log_softmax(-1)).sum(-1)
            current_time_weight = torch.where(
                current_time_target[..., -1] > .5, .1, 1.)
            current_time_mask = usable.unsqueeze(-1).expand_as(current_time_loss)
            loss = loss + args.current_event_time_weight * (
                current_time_loss * current_time_weight * current_time_mask).sum() / (
                current_time_weight * current_time_mask).sum().clamp_min(1)
            pose_mask = usable & batch["pose_mask"].squeeze(-1).bool()
            loss = loss + .2 * masked_smooth_l1(output["position"], batch["pose"], pose_mask)
            orientation_mask = usable & batch["orientation_mask"]
            loss = loss + .5 * masked_smooth_l1(
                output["orientation_6d"], batch["orientation"], orientation_mask)
            added, detail = future_loss(output, batch, usable, args)
            loss = loss + added
            if extension is not None:
                extra, extra_detail = extension.loss(model, output, batch, usable, args)
                loss = loss + extra
                detail.update(extra_detail)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
            components.append(detail)
        validation_predictions = predict(model, validation, stats, args)
        report = intent_metrics(validation_predictions, HORIZONS_MS)
        thresholds = [max([.2, .35, .5, .65, .8], key=lambda threshold:
            base.event_summary(validation_predictions, event, threshold,
                               args.uncertainty_ms / 1000)["f1"])
            for event in range(2)]
        current_report = base.metrics(
            validation_predictions, thresholds, args.uncertainty_ms / 1000)
        selection = (report["mean_future_position_cm"]
                     + .1 * report["mean_future_orientation_deg"]
                     + 5 * (1 - report["intent_mean_average_precision"]))
        print(f"epoch={epoch} loss={np.mean(losses):.4f} score={selection:.3f} "
              f"future={report['mean_future_position_cm']:.2f}cm "
              f"orientation={report['mean_future_orientation_deg']:.1f}deg "
              f"intent_mAP={report['intent_mean_average_precision']:.3f}", flush=True)
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "selection_score": selection, "validation": report,
                        "current_validation": current_report,
                        "loss_components": {k: float(np.mean([c[k] for c in components]))
                                            for k in components[0]},
                        "event_thresholds": thresholds})
        if selection < best:
            best, stale = selection, 0
            torch.save({"format": checkpoint_format,
                "state_dict": model.state_dict(), "model_args": model_args,
                "normalization": stats, "preprocessing": settings,
                "future_horizons_ms": list(HORIZONS_MS), "validation": report,
                "current_validation": current_report,
                "event_thresholds": thresholds,
                "event_tolerance_s": args.uncertainty_ms / 1000,
                "training_options": {k: str(v) if isinstance(v, Path) else v
                                     for k, v in vars(args).items()},
                "seed": args.seed}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop after {args.patience} stale epochs", flush=True)
                break
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model = model_class(**state["model_args"]).to(args.device)
    model.load_state_dict(state["state_dict"])
    full = predict(model, test, stats, args)
    without_emg = predict(model, test, stats, args, zero_emg=True)
    without_imu = predict(model, test, stats, args, zero_imu=True)
    results = {
        "emg_imu": {"current": base.metrics(
                         full, state["event_thresholds"], state["event_tolerance_s"]),
                     "future": intent_metrics(full, HORIZONS_MS)},
        "without_emg": {"current": base.metrics(
                             without_emg, state["event_thresholds"],
                             state["event_tolerance_s"]),
                         "future": intent_metrics(without_emg, HORIZONS_MS)},
        "without_imu": {"current": base.metrics(
                             without_imu, state["event_thresholds"],
                             state["event_tolerance_s"]),
                         "future": intent_metrics(without_imu, HORIZONS_MS)},
        "protocol": {"encoder_inputs": "EMG+IMU only", "vive_role": "training labels only",
                     "split_unit": args.split_by, "future_horizons_ms": list(HORIZONS_MS),
                     "annotation_uncertainty_ms": args.uncertainty_ms}}
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    if extension is not None:
        extra_results = extension.evaluate(full, args)
        (args.output_dir / "stability.json").write_text(json.dumps(extra_results, indent=2))
    torch.save(state, args.output_dir / "final.pt")
    print("TEST RESULTS", json.dumps(results, indent=2))
    print("Checkpoint:", checkpoint)


if __name__ == "__main__":
    main()
