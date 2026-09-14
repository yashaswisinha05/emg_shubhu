#!/usr/bin/env python3
"""Write one metrics JSON for a causal future checkpoint over multiple roots."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.data.click_target import add_click_target
from emg_touch.data.gripper_state import add_gripper_state
from emg_touch.data.reach_grasp import preprocess
from emg_touch.models.reach_grasp_neuro_classifier_attention import (
    NeuroClassifierConditionedAttention,
)
from emg_touch.models.reach_grasp_architecture_baselines import ArchitectureBaseline
from emg_touch.residual_gru_inference import load_residual_gru


def packed(trial, stats, key):
    valid = trial[f"{key}_valid"]
    values = (trial[key] - np.asarray(stats[key]["mean"])) / np.asarray(stats[key]["std"])
    return np.concatenate((np.where(valid, np.clip(values, -20, 20), 0), valid), -1)


def error_summary(values, unit):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "unit": unit,
            "frames": int(len(values))}


def classification_summary(target, prediction):
    """Return frame-level state metrics, or None for an empty progress bin."""
    if not len(target):
        return None
    return {
        "accuracy": accuracy_score(target, prediction),
        "frames": len(target),
    }


def causal_velocity_sequence(position, time, valid, lookback_ms):
    """Past-only least-squares velocity for every frame."""
    position, time, valid = map(np.asarray, (position, time, valid))
    velocity = np.zeros_like(position, dtype=float)
    lookback_s = lookback_ms / 1000.
    for end in range(len(time)):
        start = np.searchsorted(time, time[end] - lookback_s, side="left")
        keep = valid[start:end + 1] & np.isfinite(position[start:end + 1]).all(1)
        if keep.sum() < 2:
            continue
        stamps = time[start:end + 1][keep]
        values = position[start:end + 1][keep]
        centered = stamps - stamps.mean()
        denominator = np.square(centered).sum()
        if denominator > 0:
            velocity[end] = (
                centered[:, None] * (values - values.mean(0))).sum(0) / denominator
    return velocity


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("format") != "reach_grasp_architecture_baseline_v1":
        return load_residual_gru(checkpoint, device)
    model = ArchitectureBaseline(
        state["architecture"], **state["model_args"]).to(device)
    model.load_state_dict(state["state_dict"])
    return model.eval().requires_grad_(False), state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--velocity-lookback-ms", type=float, default=200.,
                        help="Past-only window used by the linear-extrapolation baseline")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available()
                          else "cpu")
    model, state = load_model(args.checkpoint, device)
    motion = model.motion if isinstance(model, NeuroClassifierConditionedAttention) else model
    model_modality = getattr(motion, "modality", "emg+imu")
    predicts_state = getattr(motion, "predict_state", True)
    stats = state["normalization"]
    settings = dict(state["preprocessing"])
    settings["require_events"] = False
    settings["load_pose"] = True
    paths = sorted({path for root in args.root for path in Path(root).rglob("trial_*.csv")})
    if not paths:
        parser.error("no trial_*.csv files found under the supplied roots")

    state_truth, state_prediction = [], []
    state_progress = {
        "50-75%": {"truth": [], "prediction": []},
        "75-100%": {"truth": [], "prediction": []},
    }
    position_errors, pixel_errors, pixel_x, pixel_y = [], [], [], []
    short_future, intent_future = {}, {}
    accepted, rejected = [], {}
    for number, path in enumerate(paths, 1):
        try:
            trial = add_gripper_state(path, preprocess(path, settings), settings["gap_s"])
            try:
                add_click_target(path, trial)
            except (ValueError, KeyError):
                pass
            emg_usable = trial["emg_valid"].mean(1) >= .75
            imu_usable = trial["imu_valid"].mean(1) >= .75
            wearable = (imu_usable if model_modality == "imu"
                        else emg_usable & imu_usable)
            emg = torch.from_numpy(packed(trial, stats, "emg").astype("float32"))[None].to(device)
            imu = torch.from_numpy(packed(trial, stats, "imu").astype("float32"))[None].to(device)
            with torch.no_grad():
                output = model(emg, imu)
            probability = output["gripper_state_logits"][0].softmax(-1).cpu().numpy()
            state_valid = wearable & trial["gripper_state_valid"]
            if predicts_state:
                predicted_state = probability.argmax(-1)
                state_truth.extend(trial["gripper_state"][state_valid].tolist())
                state_prediction.extend(predicted_state[state_valid].tolist())
                progress = trial["trial_progress"]
                for name, lower, upper in (("50-75%", .5, .75),
                                           ("75-100%", .75, 1.0 + 1e-8)):
                    selected = state_valid & (progress >= lower) & (progress < upper)
                    state_progress[name]["truth"].extend(
                        trial["gripper_state"][selected].tolist())
                    state_progress[name]["prediction"].extend(
                        predicted_state[selected].tolist())

            scale = np.asarray(stats["position"]["std"])
            mean = np.asarray(stats["position"]["mean"])
            current = output["position"][0].cpu().numpy() * scale + mean
            velocity = causal_velocity_sequence(
                current, trial["time"], wearable, args.velocity_lookback_ms)
            pose_valid = wearable & trial["pose_valid"]
            position_errors.extend(
                (np.linalg.norm(current - trial["position"], axis=-1)[pose_valid] * 100).tolist())
            if "click_target" in trial and output["click"] is not None:
                click = output["click"][0].clamp(0, 1).cpu().numpy()
                delta = (click - trial["click_target"]) * trial["canvas_px"]
                pixel_errors.extend(np.linalg.norm(delta[wearable], axis=-1).tolist())
                pixel_x.extend(np.abs(delta[wearable, 0]).tolist())
                pixel_y.extend(np.abs(delta[wearable, 1]).tolist())

            short = output["future_position"][0].cpu().numpy() * scale + mean
            for horizon in range(1, short.shape[1] + 1):
                if horizon >= len(wearable):
                    break
                valid = wearable[:-horizon] & trial["pose_valid"][horizon:]
                predicted = short[:-horizon, horizon - 1]
                actual = trial["position"][horizon:]
                held = current[:-horizon]
                bucket = short_future.setdefault(
                    horizon * 10, {"model": [], "hold": [], "linear": []})
                bucket["model"].extend((np.linalg.norm(predicted - actual, axis=-1)[valid] * 100).tolist())
                bucket["hold"].extend((np.linalg.norm(held - actual, axis=-1)[valid] * 100).tolist())
                linear = current[:-horizon] + horizon * .01 * velocity[:-horizon]
                bucket["linear"].extend(
                    (np.linalg.norm(linear - actual, axis=-1)[valid] * 100).tolist())

            intent = output.get("intent_position_delta")
            if intent is not None:
                intent = intent[0].cpu().numpy()
                for index, horizon in enumerate(motion.intent_horizons_steps):
                    if horizon >= len(wearable):
                        continue
                    valid = wearable[:-horizon] & trial["pose_valid"][horizon:]
                    predicted = current[:-horizon] + intent[:-horizon, index] * scale
                    actual = trial["position"][horizon:]
                    held = current[:-horizon]
                    bucket = intent_future.setdefault(
                        horizon * 10, {"model": [], "hold": [], "linear": []})
                    bucket["model"].extend((np.linalg.norm(predicted - actual, axis=-1)[valid] * 100).tolist())
                    bucket["hold"].extend((np.linalg.norm(held - actual, axis=-1)[valid] * 100).tolist())
                    linear = current[:-horizon] + horizon * .01 * velocity[:-horizon]
                    bucket["linear"].extend(
                        (np.linalg.norm(linear - actual, axis=-1)[valid] * 100).tolist())
            accepted.append(str(path))
            print(f"[{number}/{len(paths)}] {path}", file=sys.stderr, flush=True)
        except (ValueError, KeyError, RuntimeError) as error:
            rejected[str(path)] = str(error)

    result = {
        "checkpoint": str(args.checkpoint), "roots": args.root,
        "trials": {"accepted": len(accepted), "rejected": len(rejected),
                   "rejection_reasons": rejected},
        "gripper": None,
        "current_position_cm": error_summary(position_errors, "cm"),
        "pixel": {"euclidean_px": error_summary(pixel_errors, "px"),
                  "absolute_x_px": error_summary(pixel_x, "px"),
                  "absolute_y_px": error_summary(pixel_y, "px")},
        "supervised_future_position": {},
        "intent_future_position": {},
    }
    if state_truth:
        result["gripper"] = {
            "accuracy": accuracy_score(state_truth, state_prediction),
            "macro_f1": f1_score(state_truth, state_prediction,
                                  average="macro", zero_division=0),
            "confusion_open_close": confusion_matrix(
                state_truth, state_prediction, labels=[0, 1]).tolist(),
            "frames": len(state_truth),
            "accuracy_by_trial_progress": {
                name: classification_summary(values["truth"], values["prediction"])
                for name, values in state_progress.items()
            },
        }
    for name, collection in (("supervised_future_position", short_future),
                             ("intent_future_position", intent_future)):
        for horizon, values in sorted(collection.items()):
            model_error = error_summary(values["model"], "cm")
            hold_error = error_summary(values["hold"], "cm")
            linear_error = error_summary(values["linear"], "cm")
            result[name][str(horizon)] = {
                "model": model_error, "hold_current_baseline": hold_error,
                "linear_extrapolation_baseline": linear_error,
                "mean_gain_over_hold_cm": (hold_error["mean"] - model_error["mean"]
                                           if model_error and hold_error else None),
                "mean_gain_over_linear_cm": (linear_error["mean"] - model_error["mean"]
                                             if model_error and linear_error else None),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
