from __future__ import annotations

import numpy as np

from emg_touch.physics.manipulator_ik import ThreeRManipulator


def test_base_and_p1_are_same_origin_and_ik_round_trips() -> None:
    arm = ThreeRManipulator((0.5, 0.6))
    angles = np.array([0.4, 0.25, 1.1])
    chain = arm.forward(angles)
    np.testing.assert_allclose(chain[0], np.zeros(3), atol=0.0)
    solved = arm.inverse(chain[-1], previous=angles)
    np.testing.assert_allclose(solved.chain[-1], chain[-1], atol=1e-7)
    assert not solved.was_projected


def test_unreachable_target_is_reported_and_projected() -> None:
    arm = ThreeRManipulator((0.5, 0.6))
    solved = arm.inverse(np.array([2.0, 0.0, 0.0]))
    assert solved.was_projected
    assert np.linalg.norm(solved.projected) < arm.maximum_reach
    np.testing.assert_allclose(solved.chain[-1], solved.projected, atol=1e-7)


def test_continuous_trajectory_avoids_elbow_branch_flips() -> None:
    arm = ThreeRManipulator((0.5, 0.6))
    source_angles = np.stack([
        np.linspace(0.1, 0.4, 16),
        np.linspace(0.2, 0.5, 16),
        np.linspace(1.2, 0.8, 16),
    ], axis=-1)
    points = arm.forward(source_angles)[:, -1]
    followed = arm.follow(points, initial_angles=source_angles[0])
    np.testing.assert_allclose(followed["chain"][:, -1], points, atol=1e-7)
    assert np.max(np.abs(np.diff(followed["angles"], axis=0))) < 0.2
    assert not followed["was_projected"].any()
