#!/usr/bin/env python3
"""Train and honestly evaluate a causal EMG-only grasp-onset detector."""
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
from emg_touch.models.emg_grasp_onset import EMGGraspOnsetDetector


def emg_normalization(trials):
    values = np.concatenate([trial["emg"] for trial in trials])
    valid = np.concatenate([trial["emg_valid"] for trial in trials])
    mean, std = [], []
    for channel in range(values.shape[1]):
        observed = values[valid[:, channel], channel]
        if not len(observed):
            raise ValueError(f"no valid training values for EMG channel {channel}")
        mean.append(float(observed.mean()))
        std.append(float(max(observed.std(), 1e-6)))
    return {"emg": {"mean": mean, "std": std}}


def soft_grasp_target(time, event, uncertainty_s):
    """Gaussian supervision reflecting uncertainty in the manual timestamp."""
    sigma = max(.04, uncertainty_s / 2)
    distance = np.asarray(time) - float(event)
    target = np.exp(-.5 * (distance / sigma) ** 2)
    target[np.abs(distance) > uncertainty_s] = 0.
    return target.astype("float32")


def batches(trials, stats, batch_size, device, shuffle=False):
    order = list(range(len(trials)))
    if shuffle:
        random.shuffle(order)
    mean = np.asarray(stats["emg"]["mean"])
    std = np.asarray(stats["emg"]["std"])
    for start in range(0, len(order), batch_size):
        chosen = [trials[i] for i in order[start:start + batch_size]]
        length = max(len(t["time"]) for t in chosen)
        packed = np.zeros((len(chosen), length, 16), dtype="float32")
        target = np.zeros((len(chosen), length), dtype="float32")
        valid = np.zeros((len(chosen), length), dtype=bool)
        for row, trial in enumerate(chosen):
            n = len(trial["time"])
            observed = trial["emg_valid"]
            z = np.clip((trial["emg"] - mean) / std, -20, 20)
            packed[row, :n] = np.concatenate(
                [np.where(observed, z, 0), observed], axis=1)
            valid[row, :n] = observed.mean(1) >= .75
            target[row, :n] = soft_grasp_target(
                trial["time"], trial["events"][0],
                float(trial.get("annotation_uncertainty_s", .2)))
        yield {"trials": chosen,
               "emg": torch.from_numpy(packed).to(device),
               "target": torch.from_numpy(target).to(device),
               "valid": torch.from_numpy(valid).to(device)}


def focal_event_loss(logit, target, valid, positive_weight=4., gamma=2.):
    """Soft-label focal BCE with extra weight on the rare event neighbourhood."""
    probability = logit.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = target * probability + (1 - target) * (1 - probability)
    weight = (1 + (positive_weight - 1) * target) * (1 - pt).pow(gamma)
    return (bce * weight * valid).sum() / valid.sum().clamp_min(1)


def detect(time, probability, valid, threshold, persistence_s, refractory_s=.5):
    detections, start, last = [], None, -np.inf
    above = False
    for stamp, score, good in zip(time, probability, valid):
        if not good or not np.isfinite(score):
            continue
        current = score >= threshold
        if current and not above:
            start = float(stamp)
        if not current:
            start = None
        if current and start is not None and stamp - start >= persistence_s:
            if start - last >= refractory_s:
                detections.append(start)
                last = start
            start = None
        above = current
    return detections


@torch.no_grad()
def predict(model, trials, stats, args, mode="normal"):
    model.eval()
    result = []
    source = trials
    if mode == "shuffle":
        source = trials[1:] + trials[:1]
    for b, source_batch in zip(
            batches(trials, stats, args.batch_size, args.device),
            batches(source, stats, args.batch_size, args.device)):
        emg = b["emg"]
        if mode == "zero":
            emg = torch.zeros_like(emg)
        elif mode == "shuffle":
            # Keep the evaluated trial's timeline/coverage while replacing its
            # physiological evidence with a different trial. Recordings have
            # unequal lengths, so explicitly crop/pad instead of relying on
            # coincidentally equal batch shapes.
            emg = torch.zeros_like(emg)
            frames = min(emg.shape[1], source_batch["emg"].shape[1])
            emg[:, :frames] = source_batch["emg"][:, :frames]
        probability = model(emg)["grasp_logit"].sigmoid().cpu().numpy()
        valid = b["valid"].cpu().numpy()
        for row, trial in enumerate(b["trials"]):
            n = len(trial["time"])
            result.append({"trial": trial, "probability": probability[row, :n],
                           "valid": valid[row, :n]})
    return result


def score(predictions, decoder, tolerance_s):
    tp = fp = fn = 0
    errors, minutes = [], 0.
    for item in predictions:
        trial = item["trial"]
        detections = detect(trial["time"], item["probability"], item["valid"],
            decoder["threshold"], decoder["persistence_s"])
        actual = float(trial["events"][0])
        matched = [d for d in detections if abs(d - actual) <= tolerance_s]
        if matched:
            chosen = min(matched, key=lambda d: abs(d - actual))
            errors.append((chosen - actual) * 1000)
            tp += 1
        else:
            fn += 1
        fp += len(detections) - bool(matched)
        minutes += (trial["time"][-1] - trial["time"][0]) / 60
    return {"f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "tp": tp, "fp": fp, "fn": fn,
            "false_triggers_per_min": fp / max(minutes, 1e-9),
            "matched_mae_ms": float(np.mean(np.abs(errors))) if errors else None,
            "matched_signed_latency_ms": float(np.mean(errors)) if errors else None}


def choose_decoder(predictions, tolerance_s):
    candidates = [{"threshold": threshold, "persistence_s": persistence}
                  for threshold in np.arange(.1, .91, .05)
                  for persistence in [0., .02, .04, .06, .1]]
    return max(candidates, key=lambda x: (
        score(predictions, x, tolerance_s)["f1"],
        -score(predictions, x, tolerance_s)["false_triggers_per_min"]))


def schedule_predictions(train, evaluation):
    event = float(np.median([t["events"][0] for t in train]))
    result = []
    for trial in evaluation:
        probability = np.zeros(len(trial["time"]), dtype=float)
        probability[np.argmin(np.abs(trial["time"] - event))] = 1.
        result.append({"trial": trial, "probability": probability,
                       "valid": np.ones(len(probability), dtype=bool)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/emg_grasp_onset_seed42"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-rate-hz", type=float, default=1259.4)
    parser.add_argument("--rate-hz", type=float, default=200.)
    parser.add_argument("--event-origin", choices=["auto", "start"], default="auto")
    parser.add_argument("--uncertainty-ms", type=float, default=200.)
    parser.add_argument("--selection-tolerance-ms", type=float, default=200.)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--width", type=int, default=96)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use an empty output directory")
    if min(args.epochs, args.batch_size, args.patience, args.width) < 1:
        parser.error("epochs, batch size, patience and width must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    settings = {"raw_rate_hz": args.raw_rate_hz, "rate_hz": args.rate_hz,
                "gap_s": .02, "event_origin": args.event_origin,
                "event_pulse_s": .1,
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
            trials.append(trial)
        except (ValueError, KeyError) as exc:
            rejected[str(path)] = str(exc)
    if len(trials) < 20:
        raise ValueError("need at least 20 usable trials")
    np.random.default_rng(args.seed).shuffle(trials)
    n = max(1, round(.2 * len(trials)))
    test, validation, train = trials[:n], trials[n:2*n], trials[2*n:]
    stats = emg_normalization(train)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps(
        {"accepted": len(trials), "rejected": rejected}, indent=2))
    (args.output_dir / "splits.json").write_text(json.dumps({
        "train": [t["path"] for t in train],
        "validation": [t["path"] for t in validation],
        "test": [t["path"] for t in test]}, indent=2))
    print(f"Trials: train={len(train)}, validation={len(validation)}, "
          f"test={len(test)}, rejected={len(rejected)}", flush=True)

    model = EMGGraspOnsetDetector(width=args.width).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    tolerance = args.selection_tolerance_ms / 1000
    best, stale, history = -1., 0, []
    checkpoint = args.output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in batches(train, stats, args.batch_size, args.device, True):
            optimizer.zero_grad(set_to_none=True)
            logit = model(batch["emg"])["grasp_logit"]
            loss = focal_event_loss(logit, batch["target"], batch["valid"])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
        validation_predictions = predict(model, validation, stats, args)
        decoder = choose_decoder(validation_predictions, tolerance)
        report = score(validation_predictions, decoder, tolerance)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "validation": report, "decoder": decoder})
        print(f"epoch={epoch} loss={np.mean(losses):.5f} "
              f"val_f1={report['f1']:.4f} precision={report['precision']:.3f} "
              f"recall={report['recall']:.3f} fp/min={report['false_triggers_per_min']:.2f}",
              flush=True)
        if report["f1"] > best:
            best, stale = report["f1"], 0
            torch.save({"format": "emg_grasp_onset_v1", "state_dict": model.state_dict(),
                        "model_args": {"width": args.width}, "normalization": stats,
                        "preprocessing": settings, "decoder": decoder,
                        "validation": report, "seed": args.seed}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop after {args.patience} stale epochs", flush=True)
                break
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model = EMGGraspOnsetDetector(**state["model_args"]).to(args.device)
    model.load_state_dict(state["state_dict"])
    predictions = predict(model, test, stats, args)
    zero = predict(model, test, stats, args, "zero")
    shuffled = predict(model, test, stats, args, "shuffle")
    schedule = schedule_predictions(train, test)
    schedule_decoder = {"threshold": .5, "persistence_s": 0.}
    results = {"decoder_selected_on_validation": state["decoder"],
               "by_tolerance_ms": {}, "protocol": {
                   "causal": True, "split_unit": "trial",
                   "annotation_uncertainty_ms": args.uncertainty_ms}}
    for ms in [100, 150, 200, 300, 1500]:
        results["by_tolerance_ms"][str(ms)] = {
            "emg": score(predictions, state["decoder"], ms / 1000),
            "zero_emg": score(zero, state["decoder"], ms / 1000),
            "trial_shuffled_emg": score(shuffled, state["decoder"], ms / 1000),
            "schedule_only": score(schedule, schedule_decoder, ms / 1000)}
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print("TEST RESULTS", json.dumps(results, indent=2))
    print("Checkpoint:", checkpoint)


if __name__ == "__main__":
    main()
