import numpy as np

from emg_touch.deployment_control import (CommandFilteredPredictor,
                                          FinalPoseCommandFilter)
from emg_touch.physics.confidence_se3 import ConfidenceAwareSE3Controller


IDENTITY = [1., 0., 0., 0.]


def prediction(stamp, x):
    return {
        "event": "prediction", "time_s": stamp, "valid": True,
        "position_m": {"x": x, "y": 0., "z": .5},
        "orientation_quaternion_wxyz": IDENTITY,
        "holding_probability": 0., "position_uncertainty_cm": 1.,
        "orientation_uncertainty_deg": 1., "event_time_estimate_ms": None,
    }


def test_final_pose_filter_retains_raw_pose_and_emits_portable_command():
    dynamics = ConfidenceAwareSE3Controller(
        workspace_lower=(-2., -2., -2.), workspace_upper=(2., 2., 2.),
        max_velocity_mps=.2, max_acceleration_mps2=100., default_dt_s=.1)
    final = FinalPoseCommandFilter(dynamics)
    first = final.process(prediction(0., 0.))
    second = final.process(prediction(.1, 1.))
    assert first["position_m"]["x"] == 0.
    assert second["model_position_m"]["x"] == 1.
    assert 0. < second["position_m"]["x"] < 1.
    assert second["final_pose_control"]["frame"] == "model_output"


def test_terminal_commands_converge_toward_raw_final_pose():
    dynamics = ConfidenceAwareSE3Controller(
        workspace_lower=(-2., -2., -2.), workspace_upper=(2., 2., 2.),
        max_velocity_mps=.5, max_acceleration_mps2=100., default_dt_s=.1)
    final = FinalPoseCommandFilter(dynamics)
    final.process(prediction(0., 0.))
    limited = final.process(prediction(.1, .3))
    outputs = final.settle_predictions(10)
    assert outputs
    assert outputs[-1]["event"] == "terminal_pose_command"
    assert abs(outputs[-1]["position_m"]["x"] - .3) < abs(
        limited["position_m"]["x"] - .3)


class FakePredictor:
    def __init__(self):
        self.value = prediction(0., 0.)
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def add_sample(self, time_s, emg, imu):
        self.sample = (time_s, emg, imu)

    def predict(self):
        return self.value


def test_filtered_predictor_is_drop_in_hardware_independent_wrapper():
    base = FakePredictor()
    final = FinalPoseCommandFilter(ConfidenceAwareSE3Controller(
        workspace_lower=(-2., -2., -2.), workspace_upper=(2., 2., 2.)))
    wrapped = CommandFilteredPredictor(base, final)
    wrapped.reset()
    wrapped.add_sample(0., [0.] * 4, [0.] * 24)
    result = wrapped.predict()
    assert base.reset_count == 1
    assert result["position_m"] == {"x": 0., "y": 0., "z": .5}
    np.testing.assert_allclose(result["orientation_quaternion_wxyz"], IDENTITY)
