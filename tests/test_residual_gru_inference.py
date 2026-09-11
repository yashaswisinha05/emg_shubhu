import numpy as np
import torch

from emg_touch.models.reach_grasp_residual_gru import NeuromuscularResidualGRU
from emg_touch.residual_gru_inference import ResidualGRUStream


def test_live_residual_gru_returns_short_and_long_future(tmp_path):
    model_args = {"modality": "emg+imu", "width": 16, "layers": 1,
                  "dropout": 0., "future_steps": 20, "predict_click": True,
                  "predict_final_pose": False, "intent_horizons_steps": (10, 50, 100),
                  "predict_intent_state": True}
    model = NeuromuscularResidualGRU(**model_args)
    stats = {"emg": {"mean": np.zeros(8), "std": np.ones(8)},
             "imu": {"mean": np.zeros(24), "std": np.ones(24)},
             "position": {"mean": np.zeros(3), "std": np.ones(3)}}
    checkpoint = tmp_path / "model.pt"
    torch.save({"format": "neuromuscular_residual_gru_v1",
                "state_dict": model.state_dict(), "model_args": model_args,
                "normalization": stats,
                "preprocessing": {"raw_rate_hz": 1000., "rate_hz": 100.,
                                  "gap_s": .02}}, checkpoint)
    stream = ResidualGRUStream(checkpoint, device="cpu", warmup_ms=100.)
    result = None
    for index in range(301):
        result = stream.update(index / 1000., np.zeros(4), np.zeros(24)) or result
    assert result["valid"]
    assert len(result["future_positions_m"]) == 20
    assert result["intent_horizons_ms"] == [100, 500, 1000]
    assert np.asarray(result["pixel_xy"]).shape == (2,)
