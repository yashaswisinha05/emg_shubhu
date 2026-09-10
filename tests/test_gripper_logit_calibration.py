import sys

import numpy as np
import torch

from emg_touch.models.pixel_gripper_film import GripperLogitCalibration
from emg_touch.models.reach_grasp_neuromuscular_future import (
    NeuromuscularFutureGripperPoseModel,
)
from emg_touch.neuromuscular_inference import load_neuromuscular_model
from scripts import calibrate_gripper_logits_9grid as calibration
from test_reach_grasp import frame


def small_model():
    return NeuromuscularFutureGripperPoseModel(
        modality="emg+imu", width=16, patch=4, stride=2, layers=1,
        heads=4, dropout=0., react_context=20, predict_click=True,
        predict_final_pose=True, future_steps=3).eval()


def save_checkpoint(path, model):
    stats = {"emg": {"mean": np.zeros(8), "std": np.ones(8)},
             "imu": {"mean": np.zeros(24), "std": np.ones(24)},
             "position": {"mean": np.zeros(3), "std": np.ones(3)}}
    torch.save({"format": "gripper_neuromuscular_future_v1",
        "state_dict": model.state_dict(),
        "model_args": {"modality": "emg+imu", "width": 16, "patch": 4,
            "stride": 2, "layers": 1, "heads": 4, "dropout": 0.,
            "react_context": 20, "predict_click": True,
            "predict_final_pose": True, "future_steps": 3},
        "normalization": stats,
        "preprocessing": {"raw_rate_hz": 1000., "rate_hz": 100.,
            "gap_s": .02, "event_origin": "auto", "event_pulse_s": .1,
            "require_events": False}}, path)


def test_temperature_only_preserves_class_and_other_outputs():
    base = small_model()
    calibrated = GripperLogitCalibration(base, log_temperature=.4, close_bias=0.)
    emg, imu = torch.randn(1, 20, 16), torch.randn(1, 20, 48)
    emg[..., 8:], imu[..., 24:] = 1, 1
    with torch.no_grad():
        expected, actual = base(emg, imu), calibrated(emg, imu)
    assert torch.equal(expected["gripper_state_logits"].argmax(-1),
                       actual["gripper_state_logits"].argmax(-1))
    for name in ("click", "position", "orientation_6d", "final_position",
                 "future_position"):
        torch.testing.assert_close(expected[name], actual[name])


def test_nine_grid_calibration_without_vive(tmp_path, monkeypatch):
    open_root, close_root = tmp_path / "open", tmp_path / "close"
    open_root.mkdir(); close_root.mkdir()
    for state, root in (("open", open_root), ("close", close_root)):
        for index in range(9):
            data = frame()
            # Root selection is authoritative even if this optional CSV field is
            # absent or stale. Close coordinates also have realistic small jitter.
            if state == "open":
                data["gripper_state"] = "close"
                # Calibration acquisition rate may differ from checkpoint data.
                data["sample_rate_hz_declared"] = 1777.7777777777778
            data["canvas_width_px"], data["canvas_height_px"] = 1440, 900
            jitter = .002 if state == "close" else 0.
            data["click_x_norm"] = (index % 3 + 1) / 4 + jitter
            data["click_y_norm"] = (index // 3 + 1) / 4 - jitter
            data = data.drop(columns=[name for name in data if name.startswith("VIVE_")])
            data.to_csv(root / f"trial_{index:03}.csv", index=False)
    checkpoint, output = tmp_path / "base.pt", tmp_path / "calibrated.pt"
    save_checkpoint(checkpoint, small_model())
    monkeypatch.setattr(sys, "argv", ["calibrate", "--checkpoint", str(checkpoint),
        "--open-root", str(open_root), "--close-root", str(close_root),
        "--output", str(output), "--device", "cpu", "--maximum-iterations", "2"])
    calibration.main()
    state = torch.load(output, weights_only=False)
    assert state["format"] == "gripper_logit_calibration_v1"
    assert state["cross_validation"]["calibrated"]["trial_count"] == 18
    assert state["calibration_label_source"] == "open_root/close_root"
    assert max(state["grid_match_distances"]) < .01
    loaded, _ = load_neuromuscular_model(output, "cpu")
    assert isinstance(loaded, GripperLogitCalibration)
