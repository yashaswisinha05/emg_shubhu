"""Online preprocessing and fixed-weight reach/grasp orientation inference."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfilt, sosfilt_zi

from .data.hybrid_event_decoder import apply_decoder
from .models.reach_grasp_orientation_hybrid import ReachGraspOrientationHybrid
from .models.reach_grasp_robust import RobustReachGraspModel
from .models.reach_grasp_masked_reconstruction import (
    MaskedReconstructionReachGraspModel)
from .physics.manipulator_ik import ThreeRManipulator
from .physics.rotation_6d import (matrix_to_euler_zyx_degrees,
                                  matrix_to_quaternion_numpy,
                                  rotation_6d_to_matrix)


class LiveReachGraspPreprocessor:
    """Training-matched causal preprocessing for 4 EMG + 24 IMU values."""
    def __init__(self, preprocessing):
        self.raw_rate = float(preprocessing["raw_rate_hz"])
        self.rate = float(preprocessing["rate_hz"])
        self.gap = float(preprocessing["gap_s"])
        high = min(450., self.raw_rate * .4)
        self.emg_sos = butter(4, [20., high], btype="bandpass",
                              fs=self.raw_rate, output="sos")
        self.imu_sos = butter(2, 15., fs=self.raw_rate, output="sos")
        self.short = max(1, round(.02 * self.raw_rate))
        self.long = max(1, round(.05 * self.raw_rate))
        self.reset()

    def reset(self):
        self.last_time = None
        self.next_grid = None
        self.last_raw_emg = np.zeros(4)
        self.last_raw_imu = np.zeros(24)
        self.last_seen_emg = np.full(4, -np.inf)
        self.last_seen_imu = np.full(24, -np.inf)
        self.emg_state = [None] * 4
        self.imu_state = [None] * 24
        self.emg_square = [deque(maxlen=self.long) for _ in range(4)]
        self.imu_median = [deque(maxlen=3) for _ in range(24)]
        self.latest_emg = np.zeros(8, dtype="float32")
        self.latest_imu = np.zeros(24, dtype="float32")
        self.latest_ev = np.zeros(8, dtype=bool)
        self.latest_iv = np.zeros(24, dtype=bool)
        self.grid_time, self.grid_emg, self.grid_imu = [], [], []
        self.grid_ev, self.grid_iv = [], []

    @property
    def frames(self):
        return len(self.grid_time)

    def _append_grid(self, stamp):
        recent = self.last_time is not None and stamp - self.last_time <= self.gap + 1e-9
        self.grid_time.append(float(stamp))
        self.grid_emg.append(self.latest_emg.copy())
        self.grid_imu.append(self.latest_imu.copy())
        self.grid_ev.append(self.latest_ev.copy() & recent)
        self.grid_iv.append(self.latest_iv.copy() & recent)

    @staticmethod
    def _values(values, width, name):
        result = np.asarray(values, dtype=float)
        if result.shape != (width,):
            raise ValueError(f"{name} must contain exactly {width} values")
        return result

    def add_sample(self, time_s, emg, imu):
        stamp = float(time_s)
        if not np.isfinite(stamp) or (self.last_time is not None and stamp <= self.last_time):
            raise ValueError("time_s must be finite and strictly increasing")
        raw_emg = self._values(emg, 4, "emg")
        raw_imu = self._values(imu, 24, "imu")
        if self.next_grid is None:
            self.next_grid = stamp
        # Grids strictly before this sample use the preceding raw observation.
        while self.last_time is not None and self.next_grid < stamp - 1e-9:
            self._append_grid(self.next_grid)
            self.next_grid += 1 / self.rate
        row_gap = self.last_time is not None and stamp - self.last_time > self.gap
        finite = np.isfinite(raw_emg)
        self.last_raw_emg[finite] = raw_emg[finite]
        self.last_seen_emg[finite] = stamp
        emg_valid = stamp - self.last_seen_emg <= self.gap
        finite = np.isfinite(raw_imu)
        self.last_raw_imu[finite] = raw_imu[finite]
        self.last_seen_imu[finite] = stamp
        imu_valid = stamp - self.last_seen_imu <= self.gap
        if row_gap:
            self.emg_state, self.imu_state = [None] * 4, [None] * 24
            self.emg_square = [deque(maxlen=self.long) for _ in range(4)]
            self.imu_median = [deque(maxlen=3) for _ in range(24)]
        short_rms, long_rms = np.zeros(4), np.zeros(4)
        for channel in range(4):
            if not emg_valid[channel]:
                self.emg_state[channel] = None
                self.emg_square[channel].clear()
                continue
            if self.emg_state[channel] is None:
                self.emg_state[channel] = np.zeros((self.emg_sos.shape[0], 2))
            filtered, self.emg_state[channel] = sosfilt(
                self.emg_sos, [self.last_raw_emg[channel]], zi=self.emg_state[channel])
            self.emg_square[channel].append(float(filtered[0] ** 2))
            values = list(self.emg_square[channel])
            # scipy.lfilter's training-time moving average has zero initial
            # history and always divides by the complete window length.
            short_rms[channel] = np.sqrt(max(sum(values[-self.short:]) / self.short, 0))
            long_rms[channel] = np.sqrt(max(sum(values) / self.long, 0))
        for channel in range(24):
            if not imu_valid[channel]:
                self.imu_state[channel] = None
                self.imu_median[channel].clear()
                continue
            self.imu_median[channel].append(float(self.last_raw_imu[channel]))
            median = float(np.median(self.imu_median[channel]))
            if self.imu_state[channel] is None:
                self.imu_state[channel] = sosfilt_zi(self.imu_sos) * median
            filtered, self.imu_state[channel] = sosfilt(
                self.imu_sos, [median], zi=self.imu_state[channel])
            self.latest_imu[channel] = filtered[0]
        self.latest_emg = np.concatenate([short_rms, long_rms]).astype("float32")
        self.latest_ev = np.concatenate([emg_valid, emg_valid])
        self.latest_iv = imu_valid
        self.last_time = stamp
        while self.next_grid <= stamp + 1e-9:
            self._append_grid(self.next_grid)
            self.next_grid += 1 / self.rate

    def arrays(self):
        return (np.asarray(self.grid_time), np.asarray(self.grid_emg),
                np.asarray(self.grid_imu), np.asarray(self.grid_ev),
                np.asarray(self.grid_iv))


class LiveReachGraspPredictor:
    """EMG+IMU-only streaming predictor for every trained output head."""
    def __init__(self, checkpoint, device="cuda", warmup_ms=200.):
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        model_classes = {
            "reach_grasp_orientation_hybrid_v1": ReachGraspOrientationHybrid,
            "reach_grasp_robust_v1": RobustReachGraspModel,
            "reach_grasp_masked_reconstruction_v1": MaskedReconstructionReachGraspModel,
        }
        if state.get("format") not in model_classes:
            raise ValueError("checkpoint must come from the orientation or robust reach-grasp trainer")
        if state["model_args"].get("modality") != "emg+imu":
            raise ValueError("live inference requires the trained emg+imu checkpoint")
        if "hybrid_event_decoder" not in state:
            raise ValueError("checkpoint has no validation-frozen hybrid event decoder")
        self.model = model_classes[state["format"]](**state["model_args"]).to(self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval().requires_grad_(False)
        self.stats = state["normalization"]
        self.decoder = state["hybrid_event_decoder"]
        self.pipeline = LiveReachGraspPreprocessor(state["preprocessing"])
        self.warmup_s = float(warmup_ms) / 1000
        self.last_trigger = [-np.inf, -np.inf]

    def reset(self):
        self.pipeline.reset()
        self.last_trigger = [-np.inf, -np.inf]

    def add_sample(self, time_s, emg, imu):
        self.pipeline.add_sample(time_s, emg, imu)

    @torch.no_grad()
    def predict(self):
        time, emg, imu, ev, iv = self.pipeline.arrays()
        if not len(time) or time[-1] - time[0] < self.warmup_s:
            raise RuntimeError(f"need at least {self.warmup_s * 1000:.0f} ms of live samples")
        emg_z = np.clip((emg - self.stats["emg"]["mean"]) / self.stats["emg"]["std"], -20, 20)
        imu_z = np.clip((imu - self.stats["imu"]["mean"]) / self.stats["imu"]["std"], -20, 20)
        emg_input = np.concatenate([np.where(ev, emg_z, 0), ev], axis=1).astype("float32")
        imu_input = np.concatenate([np.where(iv, imu_z, 0), iv], axis=1).astype("float32")
        out = self.model(torch.from_numpy(emg_input)[None].to(self.device),
                         torch.from_numpy(imu_input)[None].to(self.device))
        probability = out["logits"][0].sigmoid().cpu().numpy()
        valid = (ev.mean(1) >= .75) & (iv.mean(1) >= .75)
        probability[~valid] = np.nan
        position = (out["position"][0].cpu().numpy() * self.stats["position"]["std"]
                    + self.stats["position"]["mean"])
        item = {"trial": {"time": time}, "prob": probability,
                "valid": valid, "position": position}
        decoded = apply_decoder([item], self.decoder)[0]
        triggers, trigger_times, trigger_probability = {}, {}, {}
        for event, name in enumerate(["grasp", "release"]):
            new = [stamp for stamp in decoded["detections"][event]
                   if stamp > self.last_trigger[event] + 1e-9]
            triggers[name] = bool(new)
            trigger_times[name] = new[-1] if new else None
            if new:
                detected = int(np.argmin(np.abs(time - new[-1])))
                value = decoded["prob"][detected, event + 1]
                trigger_probability[name] = (float(value)
                                             if np.isfinite(value) else None)
            else:
                trigger_probability[name] = None
            if new:
                self.last_trigger[event] = new[-1]
        if not valid[-1]:
            return {"event": "prediction", "time_s": float(time[-1]),
                    "valid": False, "holding_probability": None,
                    "grasp_probability": None, "release_probability": None,
                    "triggered": triggers, "trigger_time_s": trigger_times,
                    "trigger_probability": trigger_probability,
                    "position_m": None, "orientation_quaternion_wxyz": None,
                    "orientation_deg_zyx": None, "position_uncertainty_cm": None,
                    "orientation_uncertainty_deg": None,
                    "event_time_estimate_ms": None, "frames": int(len(time))}
        rotation = rotation_6d_to_matrix(out["orientation_6d"][0, -1]).cpu().numpy()
        yaw, pitch, roll = matrix_to_euler_zyx_degrees(rotation)
        quaternion = matrix_to_quaternion_numpy(rotation)
        position_uncertainty = None
        if "position_log_variance" in out:
            sigma = (out["position_log_variance"][0, -1].mul(.5).exp().cpu().numpy()
                     * np.asarray(self.stats["position"]["std"]))
            position_uncertainty = float(np.linalg.norm(sigma) * 100)
        orientation_uncertainty = None
        if "orientation_log_variance" in out:
            orientation_uncertainty = float(np.degrees(
                out["orientation_log_variance"][0, -1, 0].mul(.5).exp().cpu()))
        event_time = None
        if "event_time_logits" in out:
            horizon = out["event_time_logits"][0, -1].softmax(-1).cpu().numpy()
            centers_ms = np.array([0., 100., 200., 300., 400.])
            event_time = {}
            for index, name in enumerate(["grasp", "release"]):
                event_mass = max(float(horizon[index, :-1].sum()), 1e-9)
                event_time[name] = {
                    "expected_ms": float(horizon[index, :-1] @ centers_ms / event_mass),
                    "within_450ms_probability": float(event_mass),
                }
        return {"event": "prediction", "time_s": float(time[-1]),
                "valid": bool(valid[-1]), "holding_probability": float(probability[-1, 0]),
                "grasp_probability": float(decoded["prob"][-1, 1]),
                "release_probability": float(decoded["prob"][-1, 2]),
                "triggered": triggers, "trigger_time_s": trigger_times,
                "trigger_probability": trigger_probability,
                "position_m": {axis: float(value) for axis, value in zip("xyz", position[-1])},
                "orientation_quaternion_wxyz": [float(value) for value in quaternion],
                "orientation_deg_zyx": {"yaw": float(yaw), "pitch": float(pitch),
                                        "roll": float(roll)},
                "position_uncertainty_cm": position_uncertainty,
                "orientation_uncertainty_deg": orientation_uncertainty,
                "event_time_estimate_ms": event_time,
                "frames": int(len(time))}


class LiveThreeRGripperController:
    """Map predicted VIVE XYZ to continuous 3R IK and grasp-triggered jaws."""
    def __init__(self, link_lengths=(.50, .60), initial_joint_deg=(0., 20., 90.),
                 base_world=None, axis_order="xyz", axis_signs=(1., 1., 1.),
                 open_width_m=.08, closed_width_m=.015):
        if len(axis_order) != 3 or set(axis_order.lower()) != set("xyz"):
            raise ValueError("axis_order must be an xyz permutation")
        signs = np.asarray(axis_signs, dtype=float)
        if signs.shape != (3,) or not np.isfinite(signs).all() or np.any(signs == 0):
            raise ValueError("axis_signs must contain three finite nonzero values")
        if not 0 <= closed_width_m < open_width_m:
            raise ValueError("gripper widths must satisfy 0 <= closed < open")
        self.arm = ThreeRManipulator(tuple(link_lengths))
        self.initial = np.radians(np.asarray(initial_joint_deg, dtype=float))
        if self.initial.shape != (3,) or not np.isfinite(self.initial).all():
            raise ValueError("initial_joint_deg must contain three finite angles")
        self.base_world = None if base_world is None else np.asarray(base_world, dtype=float)
        if self.base_world is not None and (self.base_world.shape != (3,) or
                                             not np.isfinite(self.base_world).all()):
            raise ValueError("base_world must contain three finite VIVE coordinates")
        self.indices = np.array(["xyz".index(axis) for axis in axis_order.lower()])
        self.signs = signs
        self.widths = {"open": float(open_width_m), "closed": float(closed_width_m)}
        self.reset()

    def reset(self):
        self.previous = self.initial.copy()
        self.first_world = None
        self.gripper = "open"

    def _axes(self, vector):
        return np.asarray(vector, dtype=float)[self.indices] * self.signs

    def _requested(self, world):
        if self.base_world is not None:
            return self._axes(world - self.base_world), "measured shoulder/base"
        if self.first_world is None:
            self.first_world = world.copy()
        home = self.arm.forward(self.initial)[-1]
        return home + self._axes(world - self.first_world), "synthetic initial pose"

    def attach(self, prediction):
        """Attach robot/gripper output without changing the model prediction."""
        command = "hold"
        events = [(prediction["trigger_time_s"].get(name), name)
                  for name in ("grasp", "release")
                  if prediction["triggered"].get(name)]
        for _, name in sorted(events):
            wanted = "closed" if name == "grasp" else "open"
            command = "close" if wanted == "closed" else "open"
            self.gripper = wanted
        prediction["gripper"] = {"state": self.gripper, "command": command,
                                  "width_m": self.widths[self.gripper]}
        if not prediction["valid"] or prediction["position_m"] is None:
            prediction["manipulator"] = None
            return prediction
        world = np.array([prediction["position_m"][axis] for axis in "xyz"])
        requested, calibration = self._requested(world)
        solved = self.arm.inverse(requested, previous=self.previous)
        self.previous = solved.angles
        yaw = solved.angles[0]
        lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.])
        hand = solved.chain[-1]
        half = .5 * self.widths[self.gripper]
        jaws = np.stack([hand - half * lateral, hand + half * lateral])
        prediction["manipulator"] = {
            "calibration": calibration,
            "requested_endpoint_m": requested.tolist(),
            "projected_endpoint_m": solved.projected.tolist(),
            "workspace_projected": bool(solved.was_projected),
            "joint_angles_deg": {name: float(value) for name, value in zip(
                ["q1_yaw", "q2_shoulder", "q3_elbow"], np.degrees(solved.angles))},
            "chain_m": solved.chain.tolist(), "gripper_jaw_points_m": jaws.tolist(),
            "ik_fk_residual_cm": float(100 * np.linalg.norm(
                solved.chain[-1] - solved.projected))}
        return prediction
