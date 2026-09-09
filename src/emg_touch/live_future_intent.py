"""Streaming EMG+IMU inference for current and future reach-grasp intent."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .live_reach_grasp import LiveReachGraspPreprocessor
from .models.reach_grasp_future_intent import ReachGraspFutureIntentModel
from .models.reach_grasp_react_intent import ReactFutureIntentModel
from .physics.rotation_6d import (matrix_to_euler_zyx_degrees,
                                  matrix_to_quaternion_numpy,
                                  rotation_6d_to_matrix)


class LiveFutureIntentPredictor:
    """Causal current/future SE(3) and interaction inference from wearables."""

    def __init__(self, checkpoint, device="cuda", warmup_ms=200.,
                 control_horizon_ms=250, gripper_lookahead_ms=250,
                 holding_fallback=False):
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        state = torch.load(self.checkpoint, map_location=self.device,
                           weights_only=False)
        models = {"reach_grasp_future_intent_v1": ReachGraspFutureIntentModel,
                  "reach_grasp_react_intent_v1": ReactFutureIntentModel}
        if state.get("format") not in models:
            raise ValueError("checkpoint must be a supported reach-grasp future-intent model")
        if state["model_args"].get("modality") != "emg+imu":
            raise ValueError("future-intent deployment requires an EMG+IMU checkpoint")
        self.model = models[state["format"]](**state["model_args"]).to(self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval().requires_grad_(False)
        self.stats = state["normalization"]
        self.pipeline = LiveReachGraspPreprocessor(state["preprocessing"])
        self.horizons_ms = np.asarray(state["future_horizons_ms"], dtype=float)
        allowed = np.r_[0., self.horizons_ms]
        if float(control_horizon_ms) not in allowed:
            raise ValueError(f"control horizon must be one of {allowed.astype(int).tolist()} ms")
        if not 0 <= gripper_lookahead_ms <= self.horizons_ms[-1]:
            raise ValueError("gripper lookahead must lie inside the trained future horizon")
        self.control_horizon_ms = float(control_horizon_ms)
        self.gripper_lookahead_ms = float(gripper_lookahead_ms)
        self.holding_fallback = bool(holding_fallback)
        self.warmup_s = float(warmup_ms) / 1000

    def reset(self):
        self.pipeline.reset()

    def add_sample(self, time_s, emg, imu):
        self.pipeline.add_sample(time_s, emg, imu)

    @staticmethod
    def _pose(position, rotation_6d):
        rotation = rotation_6d_to_matrix(torch.as_tensor(rotation_6d)).numpy()
        quaternion = matrix_to_quaternion_numpy(rotation)
        yaw, pitch, roll = matrix_to_euler_zyx_degrees(rotation)
        return ({axis: float(value) for axis, value in zip("xyz", position)},
                [float(value) for value in quaternion],
                {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)})

    @torch.no_grad()
    def predict(self):
        time, emg, imu, emg_valid, imu_valid = self.pipeline.arrays()
        if not len(time) or time[-1] - time[0] < self.warmup_s:
            raise RuntimeError(f"need at least {self.warmup_s * 1000:.0f} ms of live samples")
        emg_z = np.clip((emg - self.stats["emg"]["mean"])
                        / self.stats["emg"]["std"], -20, 20)
        imu_z = np.clip((imu - self.stats["imu"]["mean"])
                        / self.stats["imu"]["std"], -20, 20)
        emg_input = np.concatenate([
            np.where(emg_valid, emg_z, 0), emg_valid], 1).astype("float32")
        imu_input = np.concatenate([
            np.where(imu_valid, imu_z, 0), imu_valid], 1).astype("float32")
        output = self.model(
            torch.from_numpy(emg_input)[None].to(self.device),
            torch.from_numpy(imu_input)[None].to(self.device))
        usable = ((emg_valid.mean(1) >= .75) & (imu_valid.mean(1) >= .75))
        current_probability = output["logits"][0, -1].sigmoid().cpu().numpy()
        future_event = output["future_event_logits"][0, -1].sigmoid().cpu().numpy()
        time_probability = output["intent_time_logits"][0, -1].softmax(-1).cpu().numpy()
        centers_ms = np.r_[0., self.horizons_ms]
        near = centers_ms <= self.gripper_lookahead_ms
        action_probability = time_probability[:, :-1][:, near].sum(1)
        expected_ms = []
        for event in range(2):
            event_mass = max(float(time_probability[event, :-1].sum()), 1e-9)
            expected_ms.append(float(time_probability[event, :-1] @ centers_ms
                                     / event_mass))

        current_position = (output["position"][0, -1].cpu().numpy()
            * np.asarray(self.stats["position"]["std"])
            + np.asarray(self.stats["position"]["mean"]))
        future_position = (output["future_position"][0, -1].cpu().numpy()
            * np.asarray(self.stats["position"]["std"])[None]
            + np.asarray(self.stats["position"]["mean"])[None])
        current_6d = output["orientation_6d"][0, -1].cpu().numpy()
        future_6d = output["future_orientation_6d"][0, -1].cpu().numpy()
        all_position = np.vstack([current_position, future_position])
        all_6d = np.vstack([current_6d, future_6d])
        selected = int(np.flatnonzero(
            np.r_[0., self.horizons_ms] == self.control_horizon_ms)[0])
        position, quaternion, angles = self._pose(
            all_position[selected], all_6d[selected])
        current_pose = self._pose(current_position, current_6d)
        future_quaternion = [self._pose(p, r)[1]
                             for p, r in zip(future_position, future_6d)]
        valid = bool(usable[-1])
        if not valid:
            position = quaternion = angles = None
        event_time = {name: {"expected_ms": expected_ms[index],
            "within_1s_probability": float(future_event[index]),
            "action_window_probability": float(action_probability[index]),
            "action_window_ms": self.gripper_lookahead_ms}
            for index, name in enumerate(["grasp", "release"])}
        return {
            "event": "future_intent_prediction", "time_s": float(time[-1]),
            "valid": valid, "frames": int(len(time)),
            "input": "4 EMG + 24 IMU only", "vive_is_model_input": False,
            "control_horizon_ms": self.control_horizon_ms,
            "position_m": position,
            "orientation_quaternion_wxyz": quaternion,
            "orientation_deg_zyx": angles,
            "current_position_m": current_pose[0] if valid else None,
            "current_orientation_quaternion_wxyz": current_pose[1] if valid else None,
            "future_horizons_ms": self.horizons_ms.astype(int).tolist(),
            "future_positions_m": future_position.tolist() if valid else None,
            "future_orientations_wxyz": future_quaternion if valid else None,
            "holding_probability_current": float(current_probability[0]),
            "holding_probability": (float(current_probability[0])
                                    if self.holding_fallback else None),
            "grasp_probability_current": float(current_probability[1]),
            "release_probability_current": float(current_probability[2]),
            "grasp_probability": float(action_probability[0]),
            "release_probability": float(action_probability[1]),
            "grasp_within_1s_probability": float(future_event[0]),
            "release_within_1s_probability": float(future_event[1]),
            "triggered": {"grasp": False, "release": False},
            "trigger_time_s": {"grasp": None, "release": None},
            "trigger_probability": {"grasp": None, "release": None},
            "event_time_estimate_ms": event_time,
            "position_uncertainty_cm": None,
            "orientation_uncertainty_deg": None,
        }
