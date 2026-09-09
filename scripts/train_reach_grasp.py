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
from emg_touch.physics.rotation_6d import orientation_errors_numpy, rotation_6d_to_matrix

MODEL_CLASS = ReachGraspModel
MODEL_FORMAT = "reach_grasp_annotation_v1"
MODEL_EXTRA_ARGS = {}


def event_time_targets(time, events, centers=(0., .1, .2, .3, .4),
                       uncertainty_s=.08, maximum_s=.45):
    """Soft causal targets for time-to-grasp/release plus a no-event class."""
    time = np.asarray(time, dtype=float)
    events = np.asarray(events, dtype=float)
    centers = np.asarray(centers, dtype=float)
    result = np.zeros((len(time), 2, len(centers) + 1), dtype="float32")
    for event in range(2):
        delta = events[event] - time
        near = (delta >= -uncertainty_s) & (delta <= maximum_s)
        score = np.exp(-.5 * ((delta[:, None] - centers) /
                              max(uncertainty_s, 1e-6)) ** 2)
        score /= np.maximum(score.sum(1, keepdims=True), 1e-12)
        result[near, event, :-1] = score[near]
        result[~near, event, -1] = 1.
    return result


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
        orientation = np.zeros((len(selected), length, 6), dtype="float32")
        orientation_mask = np.zeros((len(selected), length), dtype=bool)
        for i, t in enumerate(selected):
            n = len(t["time"])
            orientation[i, :n] = t.get("orientation", np.zeros((n, 6)))
            orientation_mask[i, :n] = t.get("orientation_valid", np.zeros(n, bool))
        b["orientation"] = torch.from_numpy(orientation).to(device)
        b["orientation_mask"] = torch.from_numpy(orientation_mask).to(device)
        event_time = np.zeros((len(selected), length, 2, 6), dtype="float32")
        for i, t in enumerate(selected):
            n = len(t["time"])
            uncertainty = float(t.get("annotation_uncertainty_s", .08))
            event_time[i, :n] = event_time_targets(
                t["time"], t["events"], uncertainty_s=max(.04, uncertainty))
        b["event_time_targets"] = torch.from_numpy(event_time).to(device)
        yield b


def geodesic_radians(predicted, target):
    predicted_matrix = rotation_6d_to_matrix(predicted)
    target_matrix = rotation_6d_to_matrix(target)
    relative = predicted_matrix.transpose(-1, -2) @ target_matrix
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2)
    return torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6))


def usable(b, modality):
    if modality == "emg+imu":
        return b["emg_usable"] & b["imu_usable"]
    return b[modality + "_usable"]


def masked_emg_input(packed, ratio, span):
    """Hide complete causal EMG time blocks and return valid hidden targets."""
    features = packed.shape[-1] // 2
    if packed.shape[-1] != 2 * features or features < 1:
        raise ValueError("packed EMG must contain values followed by validity flags")
    batch, frames, _ = packed.shape
    blocks = (frames + span - 1) // span
    selected = (torch.rand(batch, blocks, 1, device=packed.device) < ratio)
    selected = selected.repeat_interleave(span, dim=1)[:, :frames]
    observed = packed[..., features:] > .5
    hidden = selected.expand(-1, -1, features) & observed
    masked = packed.clone()
    masked[..., :features] = masked[..., :features].masked_fill(hidden, 0.)
    masked[..., features:] = masked[..., features:].masked_fill(hidden, 0.)
    return masked, packed[..., :features].detach(), hidden


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
            item = {"trial": trial, "prob": prob, "valid": mask, "position": position}
            if "orientation_6d" in out:
                item["orientation_6d"] = out["orientation_6d"][i, :n].cpu().numpy()
            if "event_time_logits" in out:
                item["event_time_probability"] = out["event_time_logits"][
                    i, :n].softmax(-1).cpu().numpy()
            if "position_log_variance" in out:
                item["position_std"] = (out["position_log_variance"][i, :n]
                    .mul(.5).exp().cpu().numpy() * np.asarray(stats["position"]["std"]))
            if "orientation_log_variance" in out:
                item["orientation_std_deg"] = np.degrees(
                    out["orientation_log_variance"][i, :n, 0]
                    .mul(.5).exp().cpu().numpy())
            result.append(item)
    return result


def metrics(predictions, thresholds, tolerance):
    result = {key: event_summary(predictions, e, thresholds[e], tolerance)
              for e, key in enumerate(["grasp", "release"])}
    result["event_macro_f1"] = (result["grasp"]["f1"] + result["release"]["f1"]) / 2
    tp = fp = fn = 0
    full_tp = full_fp = full_fn = 0
    certain_frames = available_frames = 0
    distances, position_sigma, position_covered = [], [], []
    orientation_errors, yaw_errors = [], []
    orientation_sigma, orientation_covered = [], []
    event_time_errors, event_horizon_correct = [], []
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
        if "position_std" in item and pm.any():
            error = np.linalg.norm(item["position"][pm] - item["trial"]["position"][pm], axis=-1)
            sigma = np.linalg.norm(item["position_std"][pm], axis=-1)
            position_sigma.extend(sigma * 100)
            position_covered.extend(error <= sigma)
        if "orientation_6d" in item:
            om = item["valid"] & item["trial"]["orientation_valid"]
            if om.any():
                geodesic, yaw = orientation_errors_numpy(
                    item["orientation_6d"][om], item["trial"]["orientation"][om])
                orientation_errors.extend(geodesic)
                yaw_errors.extend(yaw)
                if "orientation_std_deg" in item:
                    sigma = item["orientation_std_deg"][om]
                    orientation_sigma.extend(sigma)
                    orientation_covered.extend(geodesic <= sigma)
        if "event_time_probability" in item:
            centers = np.array([0., .1, .2, .3, .4])
            for event in range(2):
                delta = item["trial"]["events"][event] - item["trial"]["time"]
                considered = item["valid"] & (delta >= 0) & (delta <= .45)
                probability = item["event_time_probability"][:, event]
                conditional = probability[:, :-1] / np.maximum(
                    probability[:, :-1].sum(1, keepdims=True), 1e-9)
                expected = conditional @ centers
                event_time_errors.extend(np.abs(expected[considered] - delta[considered]) * 1000)
                event_horizon_correct.extend((probability[considered, -1] < .5).tolist())
    result["holding_f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
    result["holding_boundary_excluded"] = any("holding_certain" in i["trial"] for i in predictions)
    result["holding_f1_all_valid_frames"] = 2 * full_tp / max(1, 2 * full_tp + full_fp + full_fn)
    result["holding_certain_fraction_of_valid"] = certain_frames / max(1, available_frames)
    result["position_cm"] = float(np.mean(distances)) if distances else None
    result["valid_pose_frames"] = len(distances)
    if position_sigma:
        result["position_predicted_1sigma_cm"] = float(np.mean(position_sigma))
        result["position_1sigma_coverage"] = float(np.mean(position_covered))
    if orientation_errors:
        result["orientation_geodesic_deg"] = float(np.mean(orientation_errors))
        result["orientation_geodesic_median_deg"] = float(np.median(orientation_errors))
        result["yaw_mae_deg"] = float(np.mean(yaw_errors))
        result["yaw_median_ae_deg"] = float(np.median(yaw_errors))
        result["valid_orientation_frames"] = len(orientation_errors)
    if orientation_sigma:
        result["orientation_predicted_1sigma_deg"] = float(np.mean(orientation_sigma))
        result["orientation_1sigma_coverage"] = float(np.mean(orientation_covered))
    if event_time_errors:
        result["event_time_mae_ms_within_450ms"] = float(np.mean(event_time_errors))
        result["event_within_450ms_recall"] = float(np.mean(event_horizon_correct))
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
            for ms in [100, 150, 200, 300, 1500]}


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
    p.add_argument("--orientation-weight", type=float, default=0.,
                   help="6D rotation loss weight; active only for models with an orientation head")
    p.add_argument("--physiological-augmentation", action="store_true",
                   help="Apply causal EMG/IMU gain, placement, noise and dropout augmentation")
    p.add_argument("--augmentation-strength", type=float, default=1.)
    p.add_argument("--event-time-weight", type=float, default=0.,
                   help="Auxiliary grasp/release time-to-event distribution loss")
    p.add_argument("--pose-uncertainty-weight", type=float, default=0.,
                   help="Heteroscedastic position/orientation NLL weight")
    p.add_argument("--masked-emg-reconstruction-weight", type=float, default=0.,
                   help="Auxiliary MSE weight for causally reconstructing hidden EMG features")
    p.add_argument("--emg-reconstruction-mask-ratio", type=float, default=.35)
    p.add_argument("--emg-reconstruction-mask-span", type=int, default=5,
                   help="Contiguous hidden EMG frames per mask block")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        p.error("epochs, batch size, patience must be positive")
    if args.uncertainty_ms <= 0 or args.selection_tolerance_ms <= 0:
        p.error("uncertainty and selection tolerance must be positive")
    if min(args.orientation_weight, args.augmentation_strength,
           args.event_time_weight, args.pose_uncertainty_weight,
           args.masked_emg_reconstruction_weight) < 0:
        p.error("loss weights and augmentation strength cannot be negative")
    if not 0 < args.emg_reconstruction_mask_ratio < 1:
        p.error("EMG reconstruction mask ratio must be in (0, 1)")
    if args.emg_reconstruction_mask_span < 1:
        p.error("EMG reconstruction mask span must be positive")
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
    if args.orientation_weight and not any(t["orientation_valid"].any() for t in trials):
        raise ValueError("orientation loss requested but no valid VIVE_T0_quat_w/x/y/z labels were found")
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
    augmenter = None
    if args.physiological_augmentation:
        from emg_touch.data.wearable_augmentation import PhysiologicalWearableAugmenter
        augmenter = PhysiologicalWearableAugmenter(args.augmentation_strength)
    for modality in args.models:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        model_args = {"modality": modality, **MODEL_EXTRA_ARGS}
        model = MODEL_CLASS(**model_args).to(args.device)
        if (args.masked_emg_reconstruction_weight and modality != "imu"
                and not hasattr(model, "emg_reconstruction")):
            raise ValueError("masked EMG reconstruction requires a model reconstruction head")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        best, stale, history = float("inf"), 0, []
        path = args.output_dir / (modality.replace("+", "_") + "_best.pt")
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses, reconstruction_losses = [], []
            for b in batches(train, stats, args.batch_size, args.device, True):
                if augmenter is not None:
                    b = augmenter(b, stats, modality)
                valid = usable(b, modality)
                if not valid.any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                out = model(b["emg"], b["imu"])
                event_loss = F.binary_cross_entropy_with_logits(out["logits"], b["labels"], pos_weight=weights, reduction="none")
                loss_mask = valid.unsqueeze(-1) * b["label_mask"]
                loss = (event_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1)
                if "event_time_logits" in out and args.event_time_weight:
                    target = b["event_time_targets"]
                    horizon_loss = -(target * out["event_time_logits"].log_softmax(-1)).sum(-1)
                    # Prevent the abundant no-event frames from overwhelming
                    # the short anticipatory windows around each transition.
                    horizon_weight = torch.where(target[..., -1] > .5, .1, 1.)
                    horizon_mask = valid.unsqueeze(-1).expand_as(horizon_loss)
                    loss = loss + args.event_time_weight * (
                        horizon_loss * horizon_weight * horizon_mask).sum() / (
                        horizon_weight * horizon_mask).sum().clamp_min(1)
                if (args.masked_emg_reconstruction_weight and modality != "imu"):
                    masked_emg, emg_target, hidden = masked_emg_input(
                        b["emg"], args.emg_reconstruction_mask_ratio,
                        args.emg_reconstruction_mask_span)
                    if hidden.any():
                        reconstruction = model(masked_emg, b["imu"])[
                            "emg_reconstruction"]
                        reconstruction_loss = F.mse_loss(
                            reconstruction[hidden], emg_target[hidden])
                        loss = loss + (args.masked_emg_reconstruction_weight
                                      * reconstruction_loss)
                        reconstruction_losses.append(reconstruction_loss.item())
                pose_mask = valid & b["pose_mask"].squeeze(-1).bool()
                if pose_mask.any():
                    pose_loss = F.smooth_l1_loss(out["position"][pose_mask], b["pose"][pose_mask])
                    if "position_log_variance" in out and args.pose_uncertainty_weight:
                        residual = out["position"][pose_mask] - b["pose"][pose_mask]
                        log_variance = out["position_log_variance"][pose_mask]
                        nll = .5 * (torch.exp(-log_variance) * residual.square() + log_variance)
                        pose_loss = pose_loss + args.pose_uncertainty_weight * nll.mean()
                    loss = loss + .2 * pose_loss
                orientation_mask = valid & b["orientation_mask"]
                if "orientation_6d" in out and orientation_mask.any():
                    orientation_loss = F.smooth_l1_loss(
                        out["orientation_6d"][orientation_mask], b["orientation"][orientation_mask])
                    if "orientation_log_variance" in out and args.pose_uncertainty_weight:
                        angle = geodesic_radians(out["orientation_6d"][orientation_mask],
                                                 b["orientation"][orientation_mask])
                        log_variance = out["orientation_log_variance"][orientation_mask][:, 0]
                        nll = .5 * (torch.exp(-log_variance) * angle.square() + log_variance)
                        orientation_loss = orientation_loss + args.pose_uncertainty_weight * nll.mean()
                    loss = loss + args.orientation_weight * orientation_loss
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
            if args.orientation_weight and "orientation_geodesic_deg" in report:
                selection += .05 * report["orientation_geodesic_deg"] / 180
            reconstruction_mean = (float(np.mean(reconstruction_losses))
                                   if reconstruction_losses else None)
            print(modality, epoch, report, "masked_emg_mse=", reconstruction_mean,
                  flush=True)
            history.append({"epoch": epoch, "validation": report,
                "thresholds": thresholds, "training_loss": float(np.mean(losses)),
                "masked_emg_reconstruction_mse": reconstruction_mean})
            if selection < best:
                best, stale = selection, 0
                torch.save({"format": MODEL_FORMAT if args.annotation_aware else "reach_grasp_v1", "model_args": model_args,
                    "state_dict": model.state_dict(), "normalization": stats, "preprocessing": settings,
                    "event_thresholds": thresholds, "event_tolerance_s": tolerance, "epoch": epoch,
                    "validation": report, "seed": args.seed,
                    "orientation_weight": args.orientation_weight,
                    "robust_training": {"physiological_augmentation": args.physiological_augmentation,
                        "augmentation_strength": args.augmentation_strength,
                        "event_time_weight": args.event_time_weight,
                        "pose_uncertainty_weight": args.pose_uncertainty_weight,
                        "masked_emg_reconstruction_weight": args.masked_emg_reconstruction_weight,
                        "emg_reconstruction_mask_ratio": args.emg_reconstruction_mask_ratio,
                        "emg_reconstruction_mask_span": args.emg_reconstruction_mask_span,
                        "event_time_bin_centers_s": [0., .1, .2, .3, .4],
                        "no_event_bin": 5},
                    "orientation_representation": "6D first-two rotation-matrix columns; VIVE quaternion wxyz"}, path)
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
            model = MODEL_CLASS(**state["model_args"]).to(args.device)
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
            for e, key in enumerate(["grasp", "release"])} for ms in [100, 150, 200, 300, 1500]}
    for modality, path in selected_paths.items():
        state = torch.load(path, map_location=args.device, weights_only=False)
        model = MODEL_CLASS(**state["model_args"]).to(args.device)
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
