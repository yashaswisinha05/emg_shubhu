#!/usr/bin/env python3
"""Drive every Franka command from wearable-model outputs on a recorded trial."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.data.reach_grasp import preprocess
from emg_touch.live_reach_grasp import LiveReachGraspPredictor
from emg_touch.physics.franka_pybullet import LiveFrankaPyBulletController, PoseMapper
from scripts.live_franka_pybullet import replay_csv
from scripts.train_emg_grasp_onset import detect
from scripts.visualize_emg_grasp_franka import (
    choose_trial, grasp_probabilities)


class ModelOutputPredictor:
    """Combine full-pose/release and dedicated EMG grasp model outputs."""

    def __init__(self, pose_predictor, grasp_time_s, grasp_probability,
                 grasp_time_grid):
        self.pose_predictor = pose_predictor
        self.grasp_time_s = grasp_time_s
        self.grasp_probability = np.asarray(grasp_probability, dtype=float)
        self.grasp_time_grid = np.asarray(grasp_time_grid, dtype=float)
        self.reset()

    @property
    def pipeline(self):
        return self.pose_predictor.pipeline

    def reset(self):
        self.pose_predictor.reset()
        self.first_stamp = self.latest_stamp = None
        self.grasp_emitted = False

    def add_sample(self, stamp, emg, imu):
        stamp = float(stamp)
        if self.first_stamp is None:
            self.first_stamp = stamp
        self.latest_stamp = stamp
        self.pose_predictor.add_sample(stamp, emg, imu)

    def _attach_grasp(self, prediction):
        relative = (None if self.latest_stamp is None or self.first_stamp is None
                    else self.latest_stamp - self.first_stamp)
        probability = 0.
        if relative is not None and len(self.grasp_time_grid):
            index = int(np.clip(np.searchsorted(
                self.grasp_time_grid, relative, side="right") - 1,
                0, len(self.grasp_probability) - 1))
            probability = float(self.grasp_probability[index])
        fire = bool(self.grasp_time_s is not None and relative is not None
                    and relative >= self.grasp_time_s and not self.grasp_emitted)
        if fire:
            self.grasp_emitted = True
        # Closure follows the validation-selected EMG decoder, represented as
        # a single pulse. Raw probability remains visible for diagnostics but
        # cannot bypass persistence/threshold selection in the robot state machine.
        prediction["grasp_probability_raw_emg"] = probability
        prediction["grasp_probability"] = 1. if fire else 0.
        trigger = prediction.setdefault("trigger_probability", {})
        trigger["grasp"] = 1. if fire else None
        triggered = prediction.setdefault("triggered", {})
        triggered["grasp"] = fire
        trigger_time = prediction.setdefault("trigger_time_s", {})
        trigger_time["grasp"] = self.grasp_time_s if fire else None
        # Do not let the multitask holding head close the gripper before the
        # dedicated EMG grasp decoder fires. Release remains model-predicted.
        prediction["holding_probability"] = None
        prediction["command_sources"] = {
            "position": "EMG+IMU pose model",
            "orientation": "EMG+IMU pose model",
            "grasp": "dedicated EMG-only onset model",
            "release": "EMG+IMU interaction model"}
        prediction["vive_is_model_input"] = False
        return prediction

    def predict(self):
        return self._attach_grasp(self.pose_predictor.predict())

    def settle_predictions(self, maximum_updates):
        if not hasattr(self.pose_predictor, "settle_predictions"):
            return
        for prediction in self.pose_predictor.settle_predictions(maximum_updates):
            yield self._attach_grasp(prediction)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-checkpoint", required=True,
        help="Full EMG+IMU reach/grasp checkpoint with XYZ and orientation heads")
    parser.add_argument("--grasp-checkpoint", required=True,
        help="Dedicated emg_grasp_onset_v1 checkpoint")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--trial-csv")
    source.add_argument("--trial-root")
    parser.add_argument("--trial-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--interval-ms", type=float, default=40.)
    parser.add_argument("--warmup-ms", type=float, default=200.)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--trajectory-z-rotation-deg", type=float, default=90.)
    parser.add_argument("--home-position", type=float, nargs=3, default=(.45, 0., .50))
    parser.add_argument("--home-orientation-wxyz", type=float, nargs=4,
                        default=(0., 1., 0., 0.))
    parser.add_argument("--simulation-steps", type=int, default=24)
    parser.add_argument("--final-settle-steps", type=int, default=240)
    parser.add_argument("--release-probability-threshold", type=float, default=.9)
    args = parser.parse_args()
    if (args.speed < 0 or min(args.interval_ms, args.warmup_ms,
                              args.simulation_steps, args.final_settle_steps) <= 0):
        parser.error("speed must be nonnegative; intervals and steps must be positive")

    grasp_state = torch.load(
        args.grasp_checkpoint, map_location=args.device, weights_only=False)
    if grasp_state.get("format") != "emg_grasp_onset_v1":
        parser.error("--grasp-checkpoint is not an emg_grasp_onset_v1 checkpoint")
    try:
        trial_path, trial_source = choose_trial(
            args.grasp_checkpoint, args.trial_csv, args.trial_root, args.trial_seed)
        trial = preprocess(trial_path, grasp_state["preprocessing"])
    except ValueError as error:
        parser.error(str(error))
    probability = grasp_probabilities(grasp_state, trial, args.device)
    usable = trial["emg_valid"].mean(1) >= .75
    detections = detect(trial["time"], probability, usable,
        grasp_state["decoder"]["threshold"],
        grasp_state["decoder"]["persistence_s"])
    grasp_time = detections[0] if detections else None

    pose = LiveReachGraspPredictor(
        args.pose_checkpoint, args.device, args.warmup_ms)
    predictor = ModelOutputPredictor(
        pose, grasp_time, probability, trial["time"])
    mapper = PoseMapper(home_position=args.home_position,
        home_quaternion_wxyz=args.home_orientation_wxyz,
        trajectory_z_rotation_deg=args.trajectory_z_rotation_deg)
    try:
        controller = LiveFrankaPyBulletController(
            gui=not args.headless, mapper=mapper,
            simulation_steps=args.simulation_steps,
            grasp_probability_threshold=1.,
            release_probability_threshold=args.release_probability_threshold,
            debug_legend=("BLACK: withheld VIVE comparison   "
                          "CYAN: EMG+IMU model pose   ORANGE: Franka EE"))
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({
        "event": "model_output_franka_ready", "trial": str(trial_path),
        "trial_source": trial_source,
        "input": "4 EMG + 24 IMU only",
        "vive_role": "black comparison trajectory only",
        "pose_checkpoint": str(args.pose_checkpoint),
        "grasp_checkpoint": str(args.grasp_checkpoint),
        "predicted_grasp_s": grasp_time,
        "manual_grasp_s_comparison_only": float(trial["events"][0]),
        "all_robot_commands_are_model_outputs": True}, separators=(",", ":")),
        flush=True)
    try:
        replay_csv(predictor, controller, trial_path,
            args.interval_ms / 1000, args.speed, args.final_settle_steps)
        if not args.headless:
            print("Replay complete; close PyBullet or press Ctrl+C.", flush=True)
            try:
                while controller.p.isConnected(physicsClientId=controller.client):
                    import time
                    time.sleep(.1)
            except (KeyboardInterrupt, AttributeError):
                pass
    finally:
        controller.close()


if __name__ == "__main__":
    main()

