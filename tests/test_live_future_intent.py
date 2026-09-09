import numpy as np
import pandas as pd
import pytest
import torch

from emg_touch.live_future_intent import LiveFutureIntentPredictor
from emg_touch.models.reach_grasp_future_intent import ReachGraspFutureIntentModel
from scripts.visualize_future_intent_franka import ReplayErrorEvaluator


def checkpoint(tmp_path):
    args = {"modality": "emg+imu", "width": 16, "patch": 4, "stride": 2,
            "layers": 1, "heads": 4, "dropout": 0., "event_time_bins": 6,
            "future_horizons_ms": (100, 250, 500, 750, 1000)}
    model = ReachGraspFutureIntentModel(**args)
    path = tmp_path / "future.pt"
    torch.save({"format": "reach_grasp_future_intent_v1",
        "model_args": args, "state_dict": model.state_dict(),
        "normalization": {
            "emg": {"mean": [0.] * 8, "std": [1.] * 8},
            "imu": {"mean": [0.] * 24, "std": [1.] * 24},
            "position": {"mean": [0., 0., 0.], "std": [1., 1., 1.]}},
        "preprocessing": {"raw_rate_hz": 1000., "rate_hz": 100., "gap_s": .02},
        "future_horizons_ms": [100, 250, 500, 750, 1000]}, path)
    return path


def test_live_future_predictor_returns_robot_and_preview_poses(tmp_path):
    predictor = LiveFutureIntentPredictor(
        checkpoint(tmp_path), "cpu", warmup_ms=50, control_horizon_ms=250,
        gripper_lookahead_ms=250)
    for index in range(80):
        predictor.add_sample(index / 1000, np.ones(4) * .01, np.ones(24))
    result = predictor.predict()
    assert result["valid"] and result["vive_is_model_input"] is False
    assert result["control_horizon_ms"] == 250
    assert len(result["future_positions_m"]) == 5
    assert len(result["future_orientations_wxyz"]) == 5
    assert result["holding_probability"] is None
    assert 0 <= result["grasp_probability"] <= 1
    assert 0 <= result["release_probability"] <= 1


def test_live_future_predictor_rejects_untrained_horizon(tmp_path):
    with pytest.raises(ValueError, match="control horizon"):
        LiveFutureIntentPredictor(checkpoint(tmp_path), "cpu",
                                  control_horizon_ms=300)


class FixedPrediction:
    pipeline = None

    def predict(self):
        return {"time_s": 1., "valid": True, "control_horizon_ms": 250.,
            "current_position_m": {"x": 0., "y": 0., "z": 0.},
            "current_orientation_quaternion_wxyz": [1., 0., 0., 0.],
            "future_horizons_ms": [100, 250],
            "future_positions_m": [[.1, 0., 0.], [.25, 0., 0.]],
            "future_orientations_wxyz": [[1., 0., 0., 0.], [1., 0., 0., 0.]]}


def test_replay_error_evaluator_matches_each_future_timestamp(tmp_path):
    path = tmp_path / "trial.csv"
    frame = pd.DataFrame({"time_perf_counter": [1., 1.1, 1.25],
        "VIVE_T0_pos_x_m": [0., .2, .30], "VIVE_T0_pos_y_m": 0.,
        "VIVE_T0_pos_z_m": 0., "VIVE_T0_quat_w": 1.,
        "VIVE_T0_quat_x": 0., "VIVE_T0_quat_y": 0., "VIVE_T0_quat_z": 0.})
    frame.to_csv(path, index=False)
    result = ReplayErrorEvaluator(FixedPrediction(), path).predict()
    assert result["model_vs_vive_error_by_horizon"]["current"]["euclidean_cm"] == 0
    assert result["model_vs_vive_error_by_horizon"]["100"]["euclidean_cm"] == pytest.approx(10)
    assert result["control_horizon_model_vs_vive_error"]["euclidean_cm"] == pytest.approx(5)
    assert result["control_horizon_model_vs_vive_error"]["angle_deg"] == 0
    assert isinstance(result["control_horizon_vive_pose_comparison_only"]["position_m"], list)
