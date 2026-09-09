#!/usr/bin/env python3
"""Drive Franka with rolling EMG+IMU future pose and interaction intent."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.live_future_intent import LiveFutureIntentPredictor
from emg_touch.physics.franka_pybullet import LiveFrankaPyBulletController, PoseMapper
from scripts.live_franka_pybullet import choose_trial, replay_csv


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
