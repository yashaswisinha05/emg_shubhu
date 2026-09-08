import numpy as np
import pandas as pd

from scripts.live_franka_pybullet import choose_trial, replay_csv
from scripts.live_reach_grasp_orientation import EMG_NAMES, IMU_NAMES


class FakePredictor:
    def __init__(self):
        self.rows = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.rows.clear()

    def add_sample(self, stamp, emg, imu):
        self.rows.append((stamp, np.asarray(emg), np.asarray(imu)))

    def predict(self):
        if len(self.rows) < 2:
            raise RuntimeError("warmup")
        return {"event": "prediction", "valid": True,
                "position_m": {"x": 0., "y": 0., "z": 0.}}


class FakeController:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def set_reference_trajectory(self, positions):
        self.reference = np.asarray(positions)

    def attach(self, result):
        result["franka"] = "driven"
        return result


def recorded_frame():
    frame = pd.DataFrame({"time_perf_counter": [10.002, 10., 10.001, 10.001]})
    for index, name in enumerate(EMG_NAMES + IMU_NAMES):
        frame[name] = np.arange(4, dtype=float) + index
    # Poisoned VIVE data demonstrates that replay neither requires nor reads it.
    for axis, offset in zip("xyz", [0., 1., 2.]):
        frame[f"VIVE_T0_pos_{axis}_m"] = np.arange(4) + offset
    return frame


def test_csv_replay_is_chronological_wearable_only_and_handles_duplicates(tmp_path):
    path = tmp_path / "trial_007.csv"
    recorded_frame().to_csv(path, index=False)
    predictor, controller, output = FakePredictor(), FakeController(), []
    count = replay_csv(predictor, controller, path, interval_s=.001,
                       speed=0, emit_fn=output.append)
    assert predictor.reset_count == controller.reset_count == 1
    assert [row[0] for row in predictor.rows] == [10., 10.001, 10.002]
    assert all(row[1].shape == (4,) and row[2].shape == (24,)
               for row in predictor.rows)
    assert output[0]["vive_is_model_input"] is False
    assert controller.reference.shape == (3, 3)
    assert output[-1]["event"] == "replay_complete"
    assert count == output[-1]["predictions"] == 2


def test_random_trial_selection_is_recursive_and_seeded(tmp_path):
    for directory, name in [(tmp_path, "trial_002.csv"),
                            (tmp_path / "participant", "trial_001.csv")]:
        directory.mkdir(exist_ok=True)
        (directory / name).write_text("x\n1\n")
    assert choose_trial(tmp_path, 9) == choose_trial(tmp_path, 9)
