"""Analytical inverse kinematics for a shoulder-origin 3R arm.

The base and first shoulder point P1 are identical and fixed at the origin.
Joint 1 is yaw about +z. Joints 2 and 3 move in the yaw-selected vertical
plane. Link lengths are L1 (upper arm) and L2 (forearm).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _wrap_near(angle: float, reference: float) -> float:
    """Return the equivalent angle closest to a reference angle."""
    return reference + (angle - reference + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class IKResult:
    angles: np.ndarray
    requested: np.ndarray
    projected: np.ndarray
    chain: np.ndarray
    was_projected: bool


class ThreeRManipulator:
    """Position-only yaw/shoulder/elbow manipulator with analytical IK."""

    def __init__(self, link_lengths: tuple[float, float] = (0.50, 0.60)) -> None:
        lengths = np.asarray(link_lengths, dtype=np.float64)
        if lengths.shape != (2,) or not np.isfinite(lengths).all():
            raise ValueError("link_lengths must contain two finite values")
        if np.any(lengths <= 0.0):
            raise ValueError("link lengths must be positive")
        self.link_lengths = lengths

    @property
    def maximum_reach(self) -> float:
        return float(self.link_lengths.sum())

    @property
    def minimum_reach(self) -> float:
        return float(abs(self.link_lengths[0] - self.link_lengths[1]))

    def forward(self, angles: np.ndarray) -> np.ndarray:
        """Return base/P1, elbow, and end-effector points (..., 3, 3)."""
        q = np.asarray(angles, dtype=np.float64)
        if q.shape[-1] != 3:
            raise ValueError("angles must end in [q1, q2, q3]")
        q1, q2, q3 = np.moveaxis(q, -1, 0)
        l1, l2 = self.link_lengths
        direction_x = np.cos(q1)
        direction_y = np.sin(q1)
        elbow_radius = l1 * np.cos(q2)
        elbow = np.stack(
            [
                elbow_radius * direction_x,
                elbow_radius * direction_y,
                l1 * np.sin(q2),
            ],
            axis=-1,
        )
        hand_radius = elbow_radius + l2 * np.cos(q2 + q3)
        hand = np.stack(
            [
                hand_radius * direction_x,
                hand_radius * direction_y,
                l1 * np.sin(q2) + l2 * np.sin(q2 + q3),
            ],
            axis=-1,
        )
        base = np.zeros_like(hand)
        return np.stack([base, elbow, hand], axis=-2)

    def project_to_workspace(self, point: np.ndarray) -> tuple[np.ndarray, bool]:
        """Radially project a point into the spherical reachable shell."""
        requested = np.asarray(point, dtype=np.float64)
        if requested.shape != (3,) or not np.isfinite(requested).all():
            raise ValueError("point must be a finite XYZ vector")
        radius = float(np.linalg.norm(requested))
        epsilon = 1e-7
        lower = self.minimum_reach + epsilon
        upper = self.maximum_reach - epsilon
        clipped = float(np.clip(radius, lower, upper))
        if radius < epsilon:
            projected = np.array([clipped, 0.0, 0.0], dtype=np.float64)
        else:
            projected = requested * (clipped / radius)
        return projected, not np.isclose(radius, clipped, atol=1e-8, rtol=0.0)

    def inverse(
        self,
        point: np.ndarray,
        previous: np.ndarray | None = None,
        elbow: str = "continuous",
    ) -> IKResult:
        """Solve position IK, choosing a continuous elbow branch by default."""
        if elbow not in {"continuous", "positive", "negative"}:
            raise ValueError("elbow must be continuous, positive, or negative")
        requested = np.asarray(point, dtype=np.float64)
        projected, was_projected = self.project_to_workspace(requested)
        x, y, z = projected
        radial = float(np.hypot(x, y))
        l1, l2 = self.link_lengths
        cosine_elbow = np.clip(
            (radial * radial + z * z - l1 * l1 - l2 * l2) / (2.0 * l1 * l2),
            -1.0,
            1.0,
        )
        magnitude = float(np.arccos(cosine_elbow))
        yaw = float(np.arctan2(y, x))
        candidates = []
        for q3 in (magnitude, -magnitude):
            q2 = float(
                np.arctan2(z, radial)
                - np.arctan2(l2 * np.sin(q3), l1 + l2 * np.cos(q3))
            )
            candidates.append(np.array([yaw, q2, q3], dtype=np.float64))
        if elbow == "positive":
            angles = candidates[0]
        elif elbow == "negative":
            angles = candidates[1]
        elif previous is None:
            angles = candidates[0]
        else:
            reference = np.asarray(previous, dtype=np.float64)
            adjusted = []
            for candidate in candidates:
                value = candidate.copy()
                value[0] = _wrap_near(float(value[0]), float(reference[0]))
                value[1] = _wrap_near(float(value[1]), float(reference[1]))
                value[2] = _wrap_near(float(value[2]), float(reference[2]))
                adjusted.append(value)
            angles = min(
                adjusted, key=lambda value: float(np.linalg.norm(value - reference))
            )
        chain = self.forward(angles)
        return IKResult(angles, requested, projected, chain, was_projected)

    def follow(
        self,
        points: np.ndarray,
        initial_angles: np.ndarray | None = None,
        elbow: str = "continuous",
    ) -> dict[str, np.ndarray]:
        """Solve a continuous IK trajectory and verify it by forward kinematics."""
        requested = np.asarray(points, dtype=np.float64)
        if requested.ndim != 2 or requested.shape[1] != 3:
            raise ValueError("points must have shape [steps, 3]")
        previous = (
            None if initial_angles is None
            else np.asarray(initial_angles, dtype=np.float64)
        )
        results = []
        for point in requested:
            result = self.inverse(point, previous=previous, elbow=elbow)
            results.append(result)
            previous = result.angles
        return {
            "angles": np.stack([result.angles for result in results]),
            "requested": requested.copy(),
            "projected": np.stack([result.projected for result in results]),
            "chain": np.stack([result.chain for result in results]),
            "was_projected": np.asarray(
                [result.was_projected for result in results], dtype=bool
            ),
        }
