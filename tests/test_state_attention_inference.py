import numpy as np
import torch

from emg_touch.models.reach_grasp_state_attention_v2 import StateConditionedAttentionV2
from emg_touch.state_attention_inference import StateAttentionStream


def checkpoint(path):
    model_args = {"modality": "emg+imu", "width": 16, "patch": 4, "stride": 2,
                  "layers": 1, "heads": 4, "dropout": 0., "future_steps": 3,
                  "predict_click": True, "predict_final_pose": False,
                  "react_context": 10}
    model = StateConditionedAttentionV2(**model_args)
    torch.save({"format": "gripper_state_attention_v2",
                "state_dict": model.state_dict(), "model_args": model_args,
                "normalization": {
                    "emg": {"mean": np.zeros(8), "std": np.ones(8)},
                    "imu": {"mean": np.zeros(24), "std": np.ones(24)},
                    "position": {"mean": np.zeros(3), "std": np.ones(3)}},
                "preprocessing": {"raw_rate_hz": 1000., "rate_hz": 100.,
                                  "gap_s": .02}}, path)


def test_stream_returns_all_deployment_outputs_without_vive(tmp_path):
    path = tmp_path / "model.pt"
    checkpoint(path)
    stream = StateAttentionStream(path, "cpu", warmup_ms=20., context_ms=100.)
    result = None
    for index in range(40):
        value = stream.update(index / 1000, np.ones(4), np.ones(24), (1440, 900))
        result = value if value is not None else result
    assert result["valid"] is True
    assert len(result["position_m"]) == 3
    assert len(result["pixel_xy"]) == 2
    assert len(result["future_positions_m"]) == 3
    assert set(result["emg_channel_attention"]) == {"S0", "S4", "S8", "S12"}
    assert result["gripper_state"] in {"open", "close"}


def test_hysteresis_thresholds_are_validated(tmp_path):
    path = tmp_path / "model.pt"
    checkpoint(path)
    try:
        StateAttentionStream(path, "cpu", close_threshold=.2, open_threshold=.8)
    except ValueError as error:
        assert "thresholds" in str(error)
    else:
        raise AssertionError("invalid hysteresis thresholds were accepted")
