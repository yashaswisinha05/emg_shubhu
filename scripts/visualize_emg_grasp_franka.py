#!/usr/bin/env python3
"""Visualize EMG-only grasp detection with a PyBullet Franka gripper."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.data.reach_grasp import preprocess
from emg_touch.models.emg_grasp_onset import EMGGraspOnsetDetector
from emg_touch.physics.franka_pybullet import LiveFrankaPyBulletController, PoseMapper
from emg_touch.physics.rotation_6d import matrix_to_quaternion_numpy, rotation_6d_to_matrix
from scripts.train_emg_grasp_onset import detect


def checkpoint_test_paths(checkpoint):
    split = Path(checkpoint).resolve().parent / "splits.json"
    if not split.is_file():
        return []
    return [Path(value) for value in json.loads(split.read_text()).get("test", [])
            if Path(value).is_file()]


def choose_trial(checkpoint, trial_csv, trial_root, seed):
    if trial_csv:
        path = Path(trial_csv).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"trial CSV does not exist: {path}")
        return path, "explicit unseen/replay trial"
    paths = checkpoint_test_paths(checkpoint)
    source = "checkpoint's held-out test split"
    if trial_root:
        paths = sorted(Path(trial_root).expanduser().resolve().rglob("trial_*.csv"))
        source = "requested trial root (caller must ensure it is unseen)"
    if not paths:
        raise ValueError("no trial found; pass --trial-csv or --trial-root")
    return random.Random(seed).choice(paths), source


def pack_emg(trial, stats, device):
    mean = np.asarray(stats["emg"]["mean"])
    std = np.asarray(stats["emg"]["std"])
    observed = trial["emg_valid"]
    values = np.clip((trial["emg"] - mean) / std, -20, 20)
    packed = np.concatenate([np.where(observed, values, 0), observed], axis=1)
    return torch.from_numpy(packed.astype("float32")).unsqueeze(0).to(device)


@torch.no_grad()
def grasp_probabilities(state, trial, device):
    model = EMGGraspOnsetDetector(**state["model_args"]).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model(pack_emg(trial, state["normalization"], device))[
        "grasp_logit"].sigmoid()[0].cpu().numpy()


def trial_quaternions(trial):
    matrices = rotation_6d_to_matrix(torch.from_numpy(trial["orientation"])).numpy()
    return matrix_to_quaternion_numpy(matrices)


def replay(controller, trial, probability, decoder, speed, arm_source,
           home_position, home_orientation, headless=False):
    usable = trial["emg_valid"].mean(1) >= .75
    detections = detect(trial["time"], probability, usable,
        decoder["threshold"], decoder["persistence_s"])
    detected = detections[0] if detections else None
    pose_valid = trial["pose_valid"] & trial["orientation_valid"]
    quaternions = trial_quaternions(trial)
    controller._debug_text(
        "GRASP: EMG ONLY   ARM: " + ("WITHHELD VIVE REPLAY" if arm_source == "vive"
                                     else "FIXED HOME POSE"),
        [-.35, 0., 1.25], [.1, .1, .1], 1.4)
    print(json.dumps({"event": "replay_started", "input": "EMG only",
        "vive_is_grasp_model_input": False, "arm_source": arm_source,
        "predicted_grasp_s": detected, "manual_grasp_s": float(trial["events"][0]),
        "signed_error_ms": None if detected is None else
            1000 * (detected - float(trial["events"][0]))}), flush=True)
    wall = time.monotonic()
    for index, stamp in enumerate(trial["time"]):
        if speed > 0:
            delay = stamp / speed - (time.monotonic() - wall)
            if delay > 0:
                time.sleep(delay)
        pulse = detected is not None and abs(stamp - detected) < .5 / max(
            1., float(state_rate(trial)))
        if arm_source == "vive" and pose_valid[index]:
            position = trial["position"][index]
            quaternion = quaternions[index]
            valid = True
        else:
            position = np.asarray(home_position)
            quaternion = np.asarray(home_orientation)
            valid = True
        output = controller.attach({
            "event": "prediction", "time_s": float(stamp), "valid": valid,
            "position_m": dict(zip("xyz", map(float, position))),
            "orientation_quaternion_wxyz": list(map(float, quaternion)),
            "grasp_probability": 1.0 if pulse else float(probability[index]),
            "release_probability": 0.0, "holding_probability": None,
            "trigger_probability": {"grasp": 1.0 if pulse else None,
                                    "release": None},
            "triggered": {"grasp": bool(pulse), "release": False},
            "trigger_time_s": {"grasp": detected if pulse else None,
                               "release": None}})
        if pulse or index % max(1, round(state_rate(trial) / 5)) == 0:
            print(json.dumps({"time_s": float(stamp),
                "grasp_probability": float(probability[index]),
                "gripper": output["gripper"]["state"],
                "predicted_grasp": bool(pulse)}), flush=True)
    controller.settle(240)
    if not headless:
        print("Replay complete; close the PyBullet window or press Ctrl+C.", flush=True)
        try:
            while controller.p.isConnected(physicsClientId=controller.client):
                time.sleep(.1)
        except (KeyboardInterrupt, AttributeError):
            pass
    return detected


def state_rate(trial):
    difference = np.diff(trial["time"])
    return 1 / np.median(difference[difference > 0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--trial-csv")
    source.add_argument("--trial-root")
    parser.add_argument("--trial-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--arm-source", choices=["fixed", "vive"], default="vive")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--trajectory-z-rotation-deg", type=float, default=90.)
    parser.add_argument("--home-position", type=float, nargs=3, default=(.45, 0., .50))
    parser.add_argument("--home-orientation-wxyz", type=float, nargs=4,
                        default=(0., 1., 0., 0.))
    parser.add_argument("--simulation-steps", type=int, default=8)
    args = parser.parse_args()
    if args.speed < 0 or args.simulation_steps < 1:
        parser.error("speed must be nonnegative and simulation steps positive")
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if state.get("format") != "emg_grasp_onset_v1":
        parser.error("checkpoint is not an EMG grasp-onset model")
    try:
        path, source_name = choose_trial(
            args.checkpoint, args.trial_csv, args.trial_root, args.trial_seed)
        trial = preprocess(path, state["preprocessing"])
    except ValueError as error:
        parser.error(str(error))
    probability = grasp_probabilities(state, trial, args.device)
    mapper = PoseMapper(home_position=args.home_position,
        home_quaternion_wxyz=args.home_orientation_wxyz,
        trajectory_z_rotation_deg=args.trajectory_z_rotation_deg)
    try:
        controller = LiveFrankaPyBulletController(
            gui=not args.headless, mapper=mapper,
            simulation_steps=args.simulation_steps,
            grasp_probability_threshold=1., release_probability_threshold=1.,
            debug_legend=("CYAN: recorded VIVE arm target (not model input)   "
                          "ORANGE: Franka EE   GRIPPER: EMG-only grasp"
                          if args.arm_source == "vive" else
                          "CYAN: fixed home target   ORANGE: Franka EE   "
                          "GRIPPER: EMG-only grasp"))
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"trial": str(path), "trial_source": source_name,
        "decoder": state["decoder"], "arm_source": args.arm_source}), flush=True)
    try:
        replay(controller, trial, probability, state["decoder"], args.speed,
               args.arm_source, args.home_position, args.home_orientation_wxyz,
               args.headless)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
