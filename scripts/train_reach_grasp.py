#!/usr/bin/env python3
"""Independent grasp/release and current tracker-position training + ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emg_touch.data.reach_grasp import preprocess
from emg_touch.models.reach_grasp import ReachGraspModel
from emg_touch.data.annotation_uncertainty import soften_events, holding_transitions


def normalization(trials):
    result = {}
    for key, mask_key in [("emg", "emg_valid"), ("imu", "imu_valid"), ("position", "pose_valid")]:
        x = np.concatenate([t[key] for t in trials])
        mask = np.concatenate([t[mask_key] for t in trials])
        if mask.ndim == 1:
            mask = np.broadcast_to(mask[:, None], x.shape)
        mean, std = [], []
        for c in range(x.shape[1]):
            valid = x[mask[:, c], c]
            if not len(valid):
                raise ValueError(f"no valid training values for {key} channel {c}")
            mean.append(float(valid.mean()))
            std.append(float(max(valid.std(), .01 if key == "position" else 1e-6)))
        result[key] = {"mean": mean, "std": std}
    return result


def batches(trials, stats, batch_size, device, shuffle=False):
    order = list(range(len(trials)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), batch_size):
        selected = [trials[i] for i in order[start:start + batch_size]]
        length = max(len(t["time"]) for t in selected)
        b = {"trials": selected}
        for key in ["emg", "imu"]:
            channels = len(stats[key]["mean"])
            x = np.zeros((len(selected), length, channels * 2), dtype="float32")
            usable = np.zeros((len(selected), length), dtype=bool)
            for i, t in enumerate(selected):
                n = len(t["time"])
                valid = t[key + "_valid"]
                z = (t[key] - stats[key]["mean"]) / stats[key]["std"]
                x[i, :n] = np.concatenate([np.where(valid, np.clip(z, -20, 20), 0), valid], 1)
                usable[i, :n] = valid.mean(1) >= .75
            b[key] = torch.from_numpy(x).to(device)
            b[key + "_usable"] = torch.from_numpy(usable).to(device)
        for key, width in [("labels", 3), ("label_mask", 3), ("pose", 3), ("pose_mask", 1)]:
            x = np.zeros((len(selected), length, width), dtype="float32")
            for i, t in enumerate(selected):
                n = len(t["time"])
                value = (np.column_stack([t.get("holding_certain", np.ones(n)), np.ones((n, 2))]) if key == "label_mask" else
                         np.column_stack([t["holding"], t["event_labels"]]) if key == "labels" else
                         (t["position"] - stats["position"]["mean"]) / stats["position"]["std"]
                         if key == "pose" else t["pose_valid"][:, None])
                x[i, :n] = value
            b[key] = torch.from_numpy(x).to(device)
        yield b


def usable(b, modality):
    if modality == "emg+imu":
        return b["emg_usable"] & b["imu_usable"]
    return b[modality + "_usable"]


def detect(times, probabilities, threshold, refractory=.4, valid=None):
    """Causal rising-threshold detector; unknown samples never rearm it.

    Preserve the last observed threshold state across gaps. A genuine drop and
    rise entirely inside a gap cannot be recovered from absent measurements.
    """
    detections, above, previous = [], False, -np.inf
    if valid is None:
        valid = np.ones(len(times), dtype=bool)
    for stamp, p, good in zip(times, probabilities, valid):
        if not good or not np.isfinite(p):
            continue
        current = p >= threshold
        if current and not above and stamp - previous >= refractory:
            detections.append(float(stamp))
            previous = stamp
        above = current
    return detections


def event_summary(predictions, event, threshold, tolerance):
    tp, fp, fn, latency, minutes = 0, 0, 0, [], 0.
    for item in predictions:
        actual = item["trial"]["events"][event]
        detections = (item["detections"][event] if "detections" in item else
                      detect(item["trial"]["time"], item["prob"][:, event + 1], threshold,
                             valid=item.get("valid")))
        match = [d for d in detections if abs(d - actual) <= tolerance]
        if match:
            chosen = min(match, key=lambda d: abs(d - actual))
            latency.append((chosen - actual) * 1000)
            tp += 1
        else:
            fn += 1
        fp += len(detections) - bool(match)
        minutes += (item["trial"]["time"][-1] - item["trial"]["time"][0]) / 60
    return {"f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "tp": tp, "fp": fp, "fn": fn, "false_triggers_per_min": fp / max(minutes, 1e-9),
            "matched_mae_ms": float(np.mean(np.abs(latency))) if latency else None,
            "matched_signed_latency_ms": float(np.mean(latency)) if latency else None}


@torch.no_grad()
def predict(model, trials, stats, args, zero_emg=False):
    model.eval()
    result = []
    for b in batches(trials, stats, args.batch_size, args.device):
        out = model(torch.zeros_like(b["emg"]) if zero_emg else b["emg"], b["imu"])
        # Identical evaluation coverage across modalities; no advantage from
        # excluding a different subset of difficult/missing samples.
        valid = b["emg_usable"] & b["imu_usable"]
        for i, trial in enumerate(b["trials"]):
            n = len(trial["time"])
            prob = out["logits"][i, :n].sigmoid().cpu().numpy()
            mask = valid[i, :n].cpu().numpy()
            # Missing evidence is neither an event-negative nor a release.
            # Decoders skip it; plots display a gap, not a fabricated zero.
            prob[~mask] = np.nan
            position = out["position"][i, :n].cpu().numpy() * stats["position"]["std"] + stats["position"]["mean"]
            result.append({"trial": trial, "prob": prob, "valid": mask, "position": position})
    return result


def metrics(predictions, thresholds, tolerance):
    result = {key: event_summary(predictions, e, thresholds[e], tolerance)
              for e, key in enumerate(["grasp", "release"])}
    result["event_macro_f1"] = (result["grasp"]["f1"] + result["release"]["f1"]) / 2
    tp = fp = fn = 0
    full_tp = full_fp = full_fn = 0
    certain_frames = available_frames = 0
    distances = []
    for item in predictions:
        valid = item["valid"] & item["trial"].get("holding_certain", np.ones_like(item["valid"]))
        true = item["trial"]["holding"].astype(bool)
        pred = item["prob"][:, 0] >= .5
        full = item["valid"]
        full_tp += int((pred & true & full).sum())
        full_fp += int((pred & ~true & full).sum())
        full_fn += int((~pred & true & full).sum())
        certain_frames += int(valid.sum())
        available_frames += int(full.sum())
        tp += int((pred & true & valid).sum())
        fp += int((pred & ~true & valid).sum())
        fn += int((~pred & true & valid).sum())
        pm = item["valid"] & item["trial"]["pose_valid"]
        distances.extend(np.linalg.norm(item["position"][pm] - item["trial"]["position"][pm], axis=-1) * 100)
    result["holding_f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
    result["holding_boundary_excluded"] = any("holding_certain" in i["trial"] for i in predictions)
    result["holding_f1_all_valid_frames"] = 2 * full_tp / max(1, 2 * full_tp + full_fp + full_fn)
    result["holding_certain_fraction_of_valid"] = certain_frames / max(1, available_frames)
    result["position_cm"] = float(np.mean(distances)) if distances else None
    result["valid_pose_frames"] = len(distances)
    result["valid_wearable_fraction"] = float(np.mean(np.concatenate([i["valid"] for i in predictions])))
    return result


def transition_predictions(predictions, parameters):
    return [dict(item, detections=holding_transitions(item["trial"]["time"],
        item["prob"][:, 0], item["valid"], **parameters)) for item in predictions]


def choose_transition(predictions, tolerance):
    candidates = [{"low": low, "high": high, "persistence_s": persistence}
                  for low, high in [(.3, .7), (.4, .6)] for persistence in [.03, .06, .1]]
    return max(candidates, key=lambda parameters: metrics(
        transition_predictions(predictions, parameters), [.5, .5], tolerance)["event_macro_f1"])


def tolerance_report(predictions, thresholds, transition):
    decoded = transition_predictions(predictions, transition)
    return {str(ms): {"event_heads": metrics(predictions, thresholds, ms / 1000),
                     "holding_transitions": metrics(decoded, [.5, .5], ms / 1000)}
            for ms in [100, 150, 200, 300]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/home/nahar3/shubham/emg_shubhu/data/1dc1acaa5827")
    p.add_argument("--output-dir", type=Path, default=Path("runs/reach_grasp_seed42"))
    p.add_argument("--models", nargs="+", choices=["emg", "imu", "emg+imu"], default=["imu", "emg", "emg+imu"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw-rate-hz", type=float, default=1259.4)
    p.add_argument("--event-origin", choices=["auto", "start"], default="auto")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--annotation-aware", action="store_true")
    p.add_argument("--uncertainty-ms", type=float, default=200.)
    p.add_argument("--selection-tolerance-ms", type=float, default=200.)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        p.error("epochs, batch size, patience must be positive")
    if args.uncertainty_ms <= 0 or args.selection_tolerance_ms <= 0:
        p.error("uncertainty and selection tolerance must be positive")
    if args.annotation_aware and args.output_dir == Path("runs/reach_grasp_seed42"):
        args.output_dir = Path("runs/reach_grasp_annotation_seed42")
    tolerance = args.selection_tolerance_ms / 1000 if args.annotation_aware else .15
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("use an empty output directory")
    settings = {"raw_rate_hz": args.raw_rate_hz, "rate_hz": 100., "gap_s": .02,
                "event_origin": args.event_origin, "event_pulse_s": .1}
    if args.annotation_aware:
        settings["annotation_uncertainty_s"] = args.uncertainty_ms / 1000
    files = sorted(Path(args.root).rglob("trial_*.csv"))
    if not files:
        p.error("no trial_*.csv under root")
    trials, rejected, hashes = [], {}, {}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            rejected[str(path)] = "byte-identical duplicate of " + hashes[digest]
            continue
        hashes[digest] = str(path)
        try:
            trial = preprocess(path, settings)
            if args.annotation_aware:
                soften_events(trial, args.uncertainty_ms / 1000)
            trials.append(trial)
        except (ValueError, KeyError) as exc:
            rejected[str(path)] = str(exc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps({"rejected": rejected,
        "accepted": {t["path"]: t["audit"] for t in trials}}, indent=2))
    if len(trials) < 20:
        raise ValueError("need at least 20 usable trials; read data_audit.json for rejected labels/data")
    # Split independent trials, never randomly divide overlapping signal windows.
    np.random.default_rng(args.seed).shuffle(trials)
    n = max(1, round(.2 * len(trials)))
    test, validation, train = trials[:n], trials[n:2*n], trials[2*n:]
    stats = normalization(train)
    (args.output_dir / "splits.json").write_text(json.dumps({k: [t["path"] for t in v]
        for k, v in [("train", train), ("validation", validation), ("test", test)]}, indent=2))
    print(f"Trials: train={len(train)}, validation={len(validation)}, test={len(test)}, rejected={len(rejected)}", flush=True)
    labels = np.concatenate([np.column_stack([t["holding"], t["event_labels"]]) for t in train])
    label_mask = np.concatenate([np.column_stack([t.get("holding_certain", np.ones(len(t["time"]))),
                                                 np.ones((len(t["time"]), 2))]) for t in train])
    positive = (labels * label_mask).sum(0)
    weights = torch.tensor(np.clip((label_mask.sum(0) - positive) / np.maximum(positive, 1), 1, 50),
                           dtype=torch.float32, device=args.device)
    selected_paths = {}
    for modality in args.models:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = ReachGraspModel(modality).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        best, stale, history = float("inf"), 0, []
        path = args.output_dir / (modality.replace("+", "_") + "_best.pt")
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for b in batches(train, stats, args.batch_size, args.device, True):
                valid = usable(b, modality)
                if not valid.any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                out = model(b["emg"], b["imu"])
                event_loss = F.binary_cross_entropy_with_logits(out["logits"], b["labels"], pos_weight=weights, reduction="none")
                loss_mask = valid.unsqueeze(-1) * b["label_mask"]
                loss = (event_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1)
                pose_mask = valid & b["pose_mask"].squeeze(-1).bool()
                if pose_mask.any():
                    loss = loss + .2 * F.smooth_l1_loss(out["position"][pose_mask], b["pose"][pose_mask])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(loss.item())
            if not losses:
                raise ValueError(f"no usable training frames for {modality}")
            predictions = predict(model, validation, stats, args)
            thresholds = [max([.2, .35, .5, .65, .8], key=lambda threshold:
                event_summary(predictions, e, threshold, tolerance)["f1"]) for e in range(2)]
            report = metrics(predictions, thresholds, tolerance)
            # Primary task is interaction; pose is reported, not used to disguise event failures.
            selection = 1 - report["event_macro_f1"] + .05 * (1 - report["holding_f1"])
            print(modality, epoch, report, flush=True)
            history.append({"epoch": epoch, "validation": report, "thresholds": thresholds})
            if selection < best:
                best, stale = selection, 0
                torch.save({"format": "reach_grasp_annotation_v1" if args.annotation_aware else "reach_grasp_v1", "model_args": {"modality": modality},
                    "state_dict": model.state_dict(), "normalization": stats, "preprocessing": settings,
                    "event_thresholds": thresholds, "event_tolerance_s": tolerance, "epoch": epoch,
                    "validation": report, "seed": args.seed}, path)
            else:
                stale += 1
            if stale >= args.patience:
                break
        selected_paths[modality] = path
        (args.output_dir / (modality.replace("+", "_") + "_history.json")).write_text(json.dumps(history, indent=2))
    results = {}
    # Freeze decoder choices from validation before any test evaluation.
    if args.annotation_aware:
        for path in selected_paths.values():
            state = torch.load(path, map_location=args.device, weights_only=False)
            model = ReachGraspModel(**state["model_args"]).to(args.device)
            model.load_state_dict(state["state_dict"])
            state["holding_decoder"] = choose_transition(predict(model, validation, stats, args), tolerance)
            torch.save(state, path)
    # Detect a protocol shortcut: fixed trial timing can predict events without
    # observing either wearable. Fit this baseline on training trials only.
    median_events = np.median(np.stack([t["events"] for t in train]), axis=0)
    schedule = []
    for trial in test:
        times = trial["time"]
        prob = np.column_stack([
            (times >= median_events[0]) & (times < median_events[1]),
            (times >= median_events[0]) & (times < median_events[0] + .1),
            (times >= median_events[1]) & (times < median_events[1] + .1)]).astype(float)
        schedule.append({"trial": trial, "prob": prob})
    results["schedule_only_baseline"] = {key: event_summary(schedule, e, .5, tolerance)
        for e, key in enumerate(["grasp", "release"])}
    if args.annotation_aware:
        results["schedule_by_tolerance"] = {str(ms): {key: event_summary(schedule, e, .5, ms / 1000)
            for e, key in enumerate(["grasp", "release"])} for ms in [100, 150, 200, 300]}
    for modality, path in selected_paths.items():
        state = torch.load(path, map_location=args.device, weights_only=False)
        model = ReachGraspModel(**state["model_args"]).to(args.device)
        model.load_state_dict(state["state_dict"])
        predictions = predict(model, test, stats, args)
        results[modality] = metrics(predictions, state["event_thresholds"], tolerance)
        if args.annotation_aware:
            results[modality]["by_tolerance_ms"] = tolerance_report(predictions,
                state["event_thresholds"], state["holding_decoder"])
        if modality == "emg+imu":
            removed = predict(model, test, stats, args, True)
            results["fusion_zero_emg"] = metrics(removed, state["event_thresholds"], tolerance)
            if args.annotation_aware:
                results["fusion_zero_emg"]["by_tolerance_ms"] = tolerance_report(removed,
                    state["event_thresholds"], state["holding_decoder"])
    if args.annotation_aware:
        results["protocol"] = {"uncertainty_ms": args.uncertainty_ms,
            "selection_tolerance_ms": args.selection_tolerance_ms,
            "note": "Timing is relative to manual annotations; matched timing error excludes missed events. Holding F1 excludes uncertain boundaries."}
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print("TEST RESULTS", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
