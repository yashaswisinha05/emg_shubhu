import numpy as np
import pytest
import torch

from emg_touch.live_future_intent import LiveFutureIntentPredictor
from emg_touch.models.reach_grasp_future_intent import ReachGraspFutureIntentModel


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
