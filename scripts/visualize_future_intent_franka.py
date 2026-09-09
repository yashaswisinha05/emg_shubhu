#!/usr/bin/env python3
"""Drive Franka with rolling EMG+IMU future pose and interaction intent."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.live_future_intent import LiveFutureIntentPredictor
from emg_touch.physics.franka_pybullet import (LiveFrankaPyBulletController, PoseMapper,
                                                quaternion_angle_degrees)
from scripts.live_franka_pybullet import choose_trial, replay_csv


class ReplayErrorEvaluator:
    """Attach synchronized VIVE errors without exposing VIVE to the model."""

    def __init__(self, predictor, trial_csv, maximum_match_ms=20.):
        self.predictor = predictor
        frame = pd.read_csv(trial_csv)
        time_name = next((name for name in ("time_perf_counter", "time_s")
                          if name in frame), None)
        position_names = [f"VIVE_T0_pos_{axis}_m" for axis in "xyz"]
        quaternion_names = [f"VIVE_T0_quat_{axis}" for axis in "wxyz"]
        missing = ([] if time_name else ["time_perf_counter/time_s"]) + [
            name for name in position_names + quaternion_names if name not in frame]
        if missing:
            raise ValueError("VIVE error evaluation needs: " + ", ".join(missing))
        time = pd.to_numeric(frame[time_name], errors="coerce").to_numpy(float)
        position = frame[position_names].apply(
            pd.to_numeric, errors="coerce").to_numpy(float)
        quaternion = frame[quaternion_names].apply(
            pd.to_numeric, errors="coerce").to_numpy(float)
        valid = (np.isfinite(time) & np.isfinite(position).all(1)
                 & np.isfinite(quaternion).all(1)
                 & (np.linalg.norm(quaternion, axis=1) > 1e-8))
        if "VIVE_T0_sync_error_ms" in frame:
            sync = pd.to_numeric(frame["VIVE_T0_sync_error_ms"],
                                 errors="coerce").to_numpy(float)
            valid &= np.isfinite(sync) & (np.abs(sync) <= 20.)
        if "VIVE_T0_tracking_age_us" in frame:
            age = pd.to_numeric(frame["VIVE_T0_tracking_age_us"],
                                errors="coerce").to_numpy(float)
            valid &= np.isfinite(age) & (np.abs(age) <= 50000.)
        order = np.argsort(time[valid], kind="stable")
        self.time = time[valid][order]
        self.position = position[valid][order]
        self.quaternion = quaternion[valid][order]
        unique = np.r_[True, np.diff(self.time) > 0]
        self.time, self.position, self.quaternion = (
            self.time[unique], self.position[unique], self.quaternion[unique])
        self.maximum_match_s = float(maximum_match_ms) / 1000
        if not len(self.time):
            raise ValueError("trial has no valid synchronized VIVE poses")

    @property
    def pipeline(self):
        return self.predictor.pipeline

    def reset(self):
        self.predictor.reset()

    def add_sample(self, time_s, emg, imu):
        self.predictor.add_sample(time_s, emg, imu)

    def _truth(self, stamp):
        right = int(np.searchsorted(self.time, stamp, side="left"))
        candidates = [index for index in (right - 1, right)
                      if 0 <= index < len(self.time)]
        if not candidates:
            return None
        index = min(candidates, key=lambda value: abs(self.time[value] - stamp))
        if abs(self.time[index] - stamp) > self.maximum_match_s:
            return None
        return self.position[index], self.quaternion[index], float(self.time[index])

    @staticmethod
    def _error(position, quaternion, truth):
        if truth is None:
            return None
        true_position, true_quaternion, matched_time = truth
        return {"euclidean_cm": float(100 * np.linalg.norm(
                    np.asarray(position) - true_position)),
                "angle_deg": quaternion_angle_degrees(
                    quaternion, true_quaternion),
                "matched_vive_time_s": matched_time}

    def predict(self):
        result = self.predictor.predict()
        stamp = result["time_s"]
        comparisons = {}
        if result["valid"]:
            current_position = np.array([
                result["current_position_m"][axis] for axis in "xyz"])
            comparisons["current"] = self._error(
                current_position,
                result["current_orientation_quaternion_wxyz"],
                self._truth(stamp))
            for horizon, position, quaternion in zip(
                    result["future_horizons_ms"], result["future_positions_m"],
                    result["future_orientations_wxyz"]):
                comparisons[str(horizon)] = self._error(
                    position, quaternion, self._truth(stamp + horizon / 1000))
        result["model_vs_vive_error_by_horizon"] = comparisons
        selected = ("current" if result["control_horizon_ms"] == 0 else
                    str(int(result["control_horizon_ms"])))
        result["control_horizon_model_vs_vive_error"] = comparisons.get(selected)
        selected_truth = (None if not result["valid"] else self._truth(
            stamp + result["control_horizon_ms"] / 1000))
        result["control_horizon_vive_pose_comparison_only"] = (
            None if selected_truth is None else {
                "position_m": selected_truth[0].tolist(),
                "orientation_wxyz": selected_truth[1].tolist(),
                "matched_time_s": selected_truth[2]})
        # Explicit audit flag: these values are attached after model inference.
        result["vive_error_evaluation_is_post_prediction"] = True
        return result


class FutureIntentFrankaController(LiveFrankaPyBulletController):
    """Franka controller with a replaceable magenta future-intent preview."""

    def reset(self):
        super().reset()
        self.future_debug_items = []
        self.future_status = -1

    def _clear_future(self):
        if hasattr(self.p, "removeUserDebugItem"):
            for item in self.future_debug_items:
                self.p.removeUserDebugItem(item, physicsClientId=self.client)
        self.future_debug_items = []

    def _future_line(self, first, second, color, width):
        if not hasattr(self.p, "addUserDebugLine"):
            return
        item = self.p.addUserDebugLine(
            np.asarray(first).tolist(), np.asarray(second).tolist(), color,
            lineWidth=width, lifeTime=0, physicsClientId=self.client)
        self.future_debug_items.append(item)

    def attach(self, prediction):
        # Anchor first-pose calibration to the model's current estimate even
        # when the commanded receding-horizon target is in the future.
        current = prediction.get("current_position_m")
        current_q = prediction.get("current_orientation_quaternion_wxyz")
        if prediction.get("valid") and current is not None and current_q is not None:
            current_vector = np.array([current[axis] for axis in "xyz"])
            self.mapper.map(current_vector, current_q)
        result = super().attach(prediction)
        truth = prediction.get("control_horizon_vive_pose_comparison_only")
        if truth is not None and "actual_ee_position_m" in result.get("franka", {}):
            mapped_position, mapped_quaternion, _ = self.mapper.map(
                truth["position_m"], truth["orientation_wxyz"])
            actual_position = np.asarray(
                result["franka"]["actual_ee_position_m"], dtype=float)
            actual_quaternion = result["franka"]["actual_ee_orientation_wxyz"]
            result["franka"]["vs_vive_future_euclidean_cm"] = float(
                100 * np.linalg.norm(actual_position - mapped_position))
            result["franka"]["vs_vive_future_angle_deg"] = (
                quaternion_angle_degrees(actual_quaternion, mapped_quaternion))
            result["franka"]["vive_future_matched_time_s"] = truth["matched_time_s"]
        self._clear_future()
        future = prediction.get("future_positions_m")
        quaternions = prediction.get("future_orientations_wxyz")
        if prediction.get("valid") and future is not None and quaternions is not None:
            raw = [np.array([current[axis] for axis in "xyz"])] + [
                np.asarray(value, dtype=float) for value in future]
            qs = [current_q] + list(quaternions)
            mapped = [self.mapper.map(position, quaternion)[0]
                      for position, quaternion in zip(raw, qs)]
            for first, second in zip(mapped, mapped[1:]):
                self._future_line(first, second, [.85, 0., .85], 3.)
            chosen = int(np.flatnonzero(np.r_[0., prediction["future_horizons_ms"]]
                         == prediction["control_horizon_ms"])[0])
            self._future_line(mapped[chosen] - np.array([.02, 0., 0.]),
                              mapped[chosen] + np.array([.02, 0., 0.]),
                              [1., 0., 1.], 6.)
        grasp = prediction.get("grasp_within_1s_probability")
        release = prediction.get("release_within_1s_probability")
        message = (f"FUTURE: G={grasp:.2f} R={release:.2f}  "
                   f"CONTROL={prediction.get('control_horizon_ms', 0):g} ms")
        self.future_status = self._debug_text(
            message, [-.30, 0., .98], [.7, 0., .7], 1.5,
            self.future_status)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trial-csv")
    source.add_argument("--trial-root")
    parser.add_argument("--trial-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--interval-ms", type=float, default=40.)
    parser.add_argument("--warmup-ms", type=float, default=200.)
    parser.add_argument("--control-horizon-ms", type=int, default=250,
                        choices=[0, 100, 250, 500, 750, 1000])
    parser.add_argument("--gripper-lookahead-ms", type=int, default=250)
    parser.add_argument("--grasp-probability-threshold", type=float, default=.9)
    parser.add_argument("--release-probability-threshold", type=float, default=.9)
    parser.add_argument("--enable-holding-fallback", action="store_true")
    parser.add_argument("--trajectory-z-rotation-deg", type=float, default=90.)
    parser.add_argument("--home-position", type=float, nargs=3, default=(.45, 0., .50))
    parser.add_argument("--home-orientation-wxyz", type=float, nargs=4,
                        default=(0., 1., 0., 0.))
    parser.add_argument("--simulation-steps", type=int, default=24)
    parser.add_argument("--final-settle-steps", type=int, default=240)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.trial_root:
        try:
            args.trial_csv = choose_trial(args.trial_root, args.trial_seed)
        except ValueError as error:
            parser.error(str(error))
    predictor = LiveFutureIntentPredictor(
        args.checkpoint, args.device, args.warmup_ms,
        args.control_horizon_ms, args.gripper_lookahead_ms,
        args.enable_holding_fallback)
    try:
        predictor = ReplayErrorEvaluator(predictor, args.trial_csv)
    except ValueError as error:
        parser.error(str(error))
    mapper = PoseMapper(home_position=args.home_position,
        home_quaternion_wxyz=args.home_orientation_wxyz,
        trajectory_z_rotation_deg=args.trajectory_z_rotation_deg)
    try:
        controller = FutureIntentFrankaController(
            gui=not args.headless, mapper=mapper,
            simulation_steps=args.simulation_steps,
            grasp_probability_threshold=args.grasp_probability_threshold,
            release_probability_threshold=args.release_probability_threshold,
            debug_legend=("BLACK: withheld VIVE   CYAN: commanded forecast   "
                          "MAGENTA: rolling 1 s intent   ORANGE: Franka EE"))
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"event": "future_intent_franka_ready",
        "trial": str(args.trial_csv), "input": "4 EMG + 24 IMU only",
        "vive_role": "black comparison trajectory only",
        "control_horizon_ms": args.control_horizon_ms,
        "gripper_lookahead_ms": args.gripper_lookahead_ms,
        "grasp_threshold": args.grasp_probability_threshold,
        "release_threshold": args.release_probability_threshold},
        separators=(",", ":")), flush=True)
    try:
        replay_csv(predictor, controller, args.trial_csv,
                   args.interval_ms / 1000, args.speed,
                   args.final_settle_steps)
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
