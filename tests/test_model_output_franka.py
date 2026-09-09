import numpy as np

from scripts.visualize_model_output_franka import ModelOutputPredictor


class FakePosePredictor:
    def __init__(self):
        self.pipeline = object()
        self.output = {"valid": True, "position_m": {"x": .4, "y": 0., "z": .5},
                       "orientation_quaternion_wxyz": [1., 0., 0., 0.],
                       "grasp_probability": .99, "release_probability": .1,
                       "holding_probability": .99}

    def reset(self):
        pass

    def add_sample(self, stamp, emg, imu):
        pass

    def predict(self):
        return dict(self.output)


def test_dedicated_emg_grasp_overrides_multitask_grasp_and_holding():
    combined = ModelOutputPredictor(
        FakePosePredictor(), .5, np.array([.1, .8, .9]), np.array([0., .5, 1.]))
    combined.add_sample(10., [0] * 4, [0] * 24)
    before = combined.predict()
    assert before["grasp_probability"] == 0
    assert before["holding_probability"] is None
    combined.add_sample(10.5, [0] * 4, [0] * 24)
    event = combined.predict()
    assert event["grasp_probability"] == 1
    assert event["triggered"]["grasp"]
    again = combined.predict()
    assert again["grasp_probability"] == 0
    assert again["release_probability"] == .1
    assert event["command_sources"]["position"] == "EMG+IMU pose model"

