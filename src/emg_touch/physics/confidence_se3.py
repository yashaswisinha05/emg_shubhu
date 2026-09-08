"""Confidence-aware causal filtering and constraints for SE(3) commands."""
from __future__ import annotations

import numpy as np


def normalized(value):
    value = np.asarray(value, dtype=float)
    norm = np.linalg.norm(value)
    if value.shape != (4,) or not np.isfinite(value).all() or norm < 1e-9:
        raise ValueError("quaternion must contain four finite wxyz values")
    return value / norm


def quaternion_angle(first, second):
    return 2 * np.arccos(np.clip(abs(np.dot(normalized(first), normalized(second))), 0, 1))


def slerp(first, second, fraction):
    first, second = normalized(first), normalized(second)
    dot = float(np.dot(first, second))
    if dot < 0:
        second, dot = -second, -dot
    fraction = float(np.clip(fraction, 0, 1))
    if dot > .9995:
        return normalized(first + fraction * (second - first))
    angle = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1 - fraction) * angle) * first +
            np.sin(fraction * angle) * second) / np.sin(angle)


def clipped_norm(vector, maximum):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= maximum or norm < 1e-12:
        return vector
    return vector * maximum / norm


class ConfidenceAwareSE3Controller:
    """Turn noisy learned poses into bounded, uncertainty-aware commands.

    Safety limits are fixed parameters. Model uncertainty can only reduce
    authority; it can never increase velocity, acceleration, or workspace.
    """
    def __init__(self, workspace_lower=(.20, -.45, .15),
                 workspace_upper=(.75, .45, .85), max_velocity_mps=.8,
                 max_acceleration_mps2=3., max_angular_velocity_degps=180.,
                 max_angular_acceleration_degps2=720.,
                 position_full_uncertainty_cm=5., position_hold_uncertainty_cm=20.,
                 orientation_full_uncertainty_deg=15.,
                 orientation_hold_uncertainty_deg=60., default_dt_s=.04,
                 imminent_event_ms=150., imminent_speed_scale=.5):
        self.lower = np.asarray(workspace_lower, dtype=float)
        self.upper = np.asarray(workspace_upper, dtype=float)
        if (self.lower.shape != (3,) or self.upper.shape != (3,)
                or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all()
                or np.any(self.lower >= self.upper)):
            raise ValueError("workspace bounds must be finite lower/upper XYZ triples")
        values = [max_velocity_mps, max_acceleration_mps2,
                  max_angular_velocity_degps, max_angular_acceleration_degps2,
                  default_dt_s, imminent_event_ms]
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("motion limits, default dt, and event horizon must be positive")
        if not 0 < imminent_speed_scale <= 1:
            raise ValueError("imminent speed scale must be in (0, 1]")
        if not 0 <= position_full_uncertainty_cm < position_hold_uncertainty_cm:
            raise ValueError("position uncertainty thresholds are invalid")
        if not 0 <= orientation_full_uncertainty_deg < orientation_hold_uncertainty_deg:
            raise ValueError("orientation uncertainty thresholds are invalid")
        self.max_velocity = float(max_velocity_mps)
        self.max_acceleration = float(max_acceleration_mps2)
        self.max_angular_velocity = np.radians(max_angular_velocity_degps)
        self.max_angular_acceleration = np.radians(max_angular_acceleration_degps2)
        self.position_uncertainty = (float(position_full_uncertainty_cm),
                                     float(position_hold_uncertainty_cm))
        self.orientation_uncertainty = (float(orientation_full_uncertainty_deg),
                                        float(orientation_hold_uncertainty_deg))
        self.default_dt = float(default_dt_s)
        self.imminent_event_ms = float(imminent_event_ms)
        self.imminent_speed_scale = float(imminent_speed_scale)
        self.reset()

    def reset(self):
        self.position = self.quaternion = self.timestamp = None
        self.velocity = np.zeros(3)
        self.angular_velocity = 0.

    @staticmethod
    def _authority(value, thresholds):
        if value is None or not np.isfinite(value):
            return 1.
        full, hold = thresholds
        return float(np.clip((hold - float(value)) / max(hold - full, 1e-9), 0, 1))

    def step(self, requested_position, requested_quaternion, timestamp_s=None,
             position_uncertainty_cm=None, orientation_uncertainty_deg=None,
             event_time_estimate_ms=None, gripper_state="open"):
        requested_position = np.asarray(requested_position, dtype=float)
        requested_quaternion = normalized(requested_quaternion)
        if requested_position.shape != (3,) or not np.isfinite(requested_position).all():
            raise ValueError("requested position must contain three finite values")
        bounded = np.clip(requested_position, self.lower, self.upper)
        workspace_projected = not np.allclose(bounded, requested_position)
        pose_authority = min(
            self._authority(position_uncertainty_cm, self.position_uncertainty),
            self._authority(orientation_uncertainty_deg, self.orientation_uncertainty))
        speed_scale, imminent = 1., None
        if isinstance(event_time_estimate_ms, dict):
            name = "grasp" if gripper_state == "open" else "release"
            estimate = event_time_estimate_ms.get(name) or {}
            expected = estimate.get("expected_ms")
            probability = estimate.get("within_450ms_probability", 0.)
            if (expected is not None and np.isfinite(expected)
                    and probability is not None and probability >= .5
                    and 0 <= expected <= self.imminent_event_ms):
                speed_scale, imminent = self.imminent_speed_scale, name
        if self.position is None:
            self.position, self.quaternion = bounded.copy(), requested_quaternion.copy()
            self.timestamp = timestamp_s
            return self.position.copy(), self.quaternion.copy(), {
                "authority": pose_authority, "speed_scale": speed_scale,
                "imminent_event": imminent, "workspace_projected": workspace_projected,
                "linear_speed_mps": 0., "angular_speed_degps": 0., "initialized": True}
        dt = self.default_dt
        if (timestamp_s is not None and self.timestamp is not None
                and np.isfinite(timestamp_s) and np.isfinite(self.timestamp)
                and timestamp_s > self.timestamp):
            dt = float(np.clip(timestamp_s - self.timestamp, .005, .2))
        desired_velocity = clipped_norm((bounded - self.position) / dt,
                                        self.max_velocity * pose_authority * speed_scale)
        acceleration = clipped_norm((desired_velocity - self.velocity) / dt,
                                    self.max_acceleration)
        self.velocity = self.velocity + acceleration * dt
        # Authority can fall abruptly, but acceleration may not: decelerate
        # toward zero at the configured physical limit instead of teleporting
        # the velocity state to zero.
        self.velocity = clipped_norm(self.velocity, self.max_velocity)
        self.position = np.clip(self.position + self.velocity * dt, self.lower, self.upper)
        angle = quaternion_angle(self.quaternion, requested_quaternion)
        wanted_angular_velocity = min(angle / dt,
            self.max_angular_velocity * pose_authority * speed_scale)
        change = np.clip(wanted_angular_velocity - self.angular_velocity,
                         -self.max_angular_acceleration * dt,
                         self.max_angular_acceleration * dt)
        self.angular_velocity = np.clip(self.angular_velocity + change,
                                        0., self.max_angular_velocity)
        step_angle = min(angle, max(0., self.angular_velocity) * dt)
        self.quaternion = slerp(self.quaternion, requested_quaternion,
                                1. if angle < 1e-9 else step_angle / angle)
        self.timestamp = timestamp_s
        return self.position.copy(), self.quaternion.copy(), {
            "authority": pose_authority, "speed_scale": speed_scale,
            "imminent_event": imminent, "workspace_projected": workspace_projected,
            "linear_speed_mps": float(np.linalg.norm(self.velocity)),
            "angular_speed_degps": float(np.degrees(self.angular_velocity)),
            "initialized": False}
