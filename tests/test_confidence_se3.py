import numpy as np

from emg_touch.physics.confidence_se3 import (
    ConfidenceAwareSE3Controller, quaternion_angle)


IDENTITY = (1., 0., 0., 0.)


def test_workspace_velocity_and_acceleration_are_bounded():
    control = ConfidenceAwareSE3Controller(
        workspace_lower=(0., -1., 0.), workspace_upper=(1., 1., 1.),
        max_velocity_mps=.2, max_acceleration_mps2=.5,
        max_angular_velocity_degps=90., max_angular_acceleration_degps2=180.,
        default_dt_s=.1)
    first, _, report = control.step((0., 0., .5), IDENTITY, 0.)
    np.testing.assert_allclose(first, (0., 0., .5))
    second, _, report = control.step((2., 0., .5), IDENTITY, .1)
    # Workspace projects x=2 to x=1. Acceleration permits only 0.05 m/s,
    # hence 5 mm displacement in the first 100 ms control interval.
    np.testing.assert_allclose(second, (.005, 0., .5), atol=1e-9)
    assert report["workspace_projected"]
    assert report["linear_speed_mps"] <= .05 + 1e-9


def test_uncertainty_only_reduces_authority_and_decelerates_safely():
    control = ConfidenceAwareSE3Controller(
        workspace_lower=(-2., -2., -2.), workspace_upper=(2., 2., 2.),
        max_velocity_mps=1., max_acceleration_mps2=1., default_dt_s=.1)
    control.step((0., 0., 0.), IDENTITY, 0.)
    control.step((1., 0., 0.), IDENTITY, .1)
    before = np.linalg.norm(control.velocity)
    _, _, report = control.step((1., 0., 0.), IDENTITY, .2,
                                position_uncertainty_cm=100.)
    assert report["authority"] == 0.
    assert np.linalg.norm(control.velocity) < before
    assert np.linalg.norm(control.velocity) >= 0.


def test_event_horizon_slows_but_does_not_change_safety_limits():
    control = ConfidenceAwareSE3Controller(
        workspace_lower=(-2., -2., -2.), workspace_upper=(2., 2., 2.),
        max_velocity_mps=1., max_acceleration_mps2=100., default_dt_s=.1,
        imminent_event_ms=150., imminent_speed_scale=.5)
    control.step((0., 0., 0.), IDENTITY, 0.)
    _, _, report = control.step((1., 0., 0.), IDENTITY, .1,
        event_time_estimate_ms={"grasp": {
            "expected_ms": 100., "within_450ms_probability": .8}},
        gripper_state="open")
    assert report["imminent_event"] == "grasp"
    assert report["speed_scale"] == .5
    assert report["linear_speed_mps"] <= .5 + 1e-9


def test_angular_rate_is_bounded():
    control = ConfidenceAwareSE3Controller(
        workspace_lower=(-1., -1., -1.), workspace_upper=(1., 1., 1.),
        max_angular_velocity_degps=30., max_angular_acceleration_degps2=60.,
        default_dt_s=.1)
    control.step((0., 0., 0.), IDENTITY, 0.)
    ninety_z = (np.sqrt(.5), 0., 0., np.sqrt(.5))
    _, command, report = control.step((0., 0., 0.), ninety_z, .1)
    assert report["angular_speed_degps"] <= 6. + 1e-9
    assert np.degrees(quaternion_angle(IDENTITY, command)) <= .6 + 1e-9
