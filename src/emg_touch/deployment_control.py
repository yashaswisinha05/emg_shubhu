"""Hardware-independent conditioning of learned SE(3) pose predictions."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .physics.confidence_se3 import (ConfidenceAwareSE3Controller,
                                     quaternion_angle)
from .physics.rotation_6d import (matrix_to_euler_zyx_degrees,
                                  quaternion_to_matrix_numpy)


class FinalPoseCommandFilter:
    """Convert raw model pose estimates into portable bounded SE(3) commands.

    The filter operates in the model output frame.  A robot-specific transform,
    workspace projection, and inverse kinematics therefore belong downstream.
    Raw estimates are retained under ``model_*`` keys for diagnostics.
    """

    def __init__(self, controller: ConfidenceAwareSE3Controller):
        self.controller = controller
        self.last_prediction = None
        self.last_requested_position = None
        self.last_requested_quaternion = None

    def reset(self):
        self.controller.reset()
        self.last_prediction = None
        self.last_requested_position = None
        self.last_requested_quaternion = None

    @staticmethod
    def _position(value):
        return np.asarray([value[axis] for axis in "xyz"], dtype=float)

    def _condition(self, prediction, requested_position, requested_quaternion,
                   timestamp_s, *, retain_raw):
        result = deepcopy(prediction)
        holding = prediction.get("holding_probability")
        inferred_gripper = ("closed" if holding is not None
                            and np.isfinite(holding) and holding >= .5 else "open")
        position, quaternion, report = self.controller.step(
            requested_position, requested_quaternion, timestamp_s,
            prediction.get("position_uncertainty_cm"),
            prediction.get("orientation_uncertainty_deg"),
            prediction.get("event_time_estimate_ms"),
            prediction.get("gripper_state", inferred_gripper))
        if retain_raw:
            result["model_position_m"] = deepcopy(prediction["position_m"])
            result["model_orientation_quaternion_wxyz"] = list(
                prediction["orientation_quaternion_wxyz"])
            result["model_orientation_deg_zyx"] = deepcopy(
                prediction.get("orientation_deg_zyx"))
        result["position_m"] = {
            axis: float(value) for axis, value in zip("xyz", position)}
        result["orientation_quaternion_wxyz"] = [
            float(value) for value in quaternion]
        yaw, pitch, roll = matrix_to_euler_zyx_degrees(
            quaternion_to_matrix_numpy(quaternion))
        result["orientation_deg_zyx"] = {
            "yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}
        result["final_pose_control"] = {
            **report,
            "frame": "model_output",
            "downstream_contract": "map XYZ+quaternion, then enforce system workspace/IK",
        }
        self.last_prediction = result
        return result

    def process(self, prediction):
        """Condition one causal prediction and preserve the raw model pose."""
        if (not prediction.get("valid") or prediction.get("position_m") is None
                or prediction.get("orientation_quaternion_wxyz") is None):
            result = deepcopy(prediction)
            result["model_position_m"] = deepcopy(prediction.get("position_m"))
            result["model_orientation_quaternion_wxyz"] = deepcopy(
                prediction.get("orientation_quaternion_wxyz"))
            result["final_pose_control"] = {
                "frame": "model_output", "held": True,
                "reason": "invalid wearable pose frame"}
            self.last_prediction = result
            return result
        requested_position = self._position(prediction["position_m"])
        requested_quaternion = np.asarray(
            prediction["orientation_quaternion_wxyz"], dtype=float)
        self.last_requested_position = requested_position.copy()
        self.last_requested_quaternion = requested_quaternion.copy()
        return self._condition(
            prediction, requested_position, requested_quaternion,
            prediction.get("time_s"), retain_raw=True)

    def settle_predictions(self, maximum_updates):
        """Generate portable terminal commands until the final request is reached."""
        if (maximum_updates <= 0 or self.last_prediction is None
                or self.last_requested_position is None):
            return []
        outputs = []
        for _ in range(int(maximum_updates)):
            timestamp = (None if self.controller.timestamp is None else
                         self.controller.timestamp + self.controller.default_dt)
            output = self._condition(
                self.last_prediction, self.last_requested_position,
                self.last_requested_quaternion, timestamp, retain_raw=False)
            output["event"] = "terminal_pose_command"
            outputs.append(output)
            position = self._position(output["position_m"])
            quaternion = np.asarray(
                output["orientation_quaternion_wxyz"], dtype=float)
            if (np.linalg.norm(position - self.last_requested_position) < 1e-3
                    and np.degrees(quaternion_angle(
                        quaternion, self.last_requested_quaternion)) < 1.):
                break
        return outputs


class CommandFilteredPredictor:
    """Apply final-pose conditioning at the predictor/system boundary."""

    def __init__(self, predictor, command_filter):
        self.predictor = predictor
        self.command_filter = command_filter

    def __getattr__(self, name):
        return getattr(self.predictor, name)

    def reset(self):
        self.predictor.reset()
        self.command_filter.reset()

    def add_sample(self, time_s, emg, imu):
        self.predictor.add_sample(time_s, emg, imu)

    def predict(self):
        return self.command_filter.process(self.predictor.predict())

    def settle_predictions(self, maximum_updates):
        return self.command_filter.settle_predictions(maximum_updates)
