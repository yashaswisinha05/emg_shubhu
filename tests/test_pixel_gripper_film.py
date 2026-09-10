import numpy as np
import torch
import sys

from emg_touch.models.pixel_gripper_film import PixelGripperFiLM
from emg_touch.models.reach_grasp_neuromuscular_future import (
    NeuromuscularFutureGripperPoseModel,
)
from emg_touch.neuromuscular_inference import NeuromuscularStream
from scripts import calibrate_pixel_gripper_film as calibration
from test_reach_grasp import frame


def model():
    return NeuromuscularFutureGripperPoseModel(
        modality="emg+imu", width=16, patch=4, stride=2, layers=1,
        heads=4, dropout=0., react_context=20, predict_click=True,
        predict_final_pose=True, future_steps=3).eval()


def inputs(length=24):
    emg, imu = torch.randn(1, length, 16), torch.randn(1, length, 48)
    emg[..., 8:], imu[..., 24:] = 1, 1
    return emg, imu


def test_identity_adapter_changes_only_calibrated_outputs():
    base = model()
    emg, imu = inputs()
    adapter = PixelGripperFiLM(base, groups=4)
    with torch.no_grad():
        expected = base(emg, imu)
        adapted = adapter(emg, imu)
    torch.testing.assert_close(adapted["click"], expected["click"])
    torch.testing.assert_close(adapted["gripper_state_logits"],
                               expected["gripper_state_logits"])
    for name in ("position", "orientation_6d", "final_position",
                 "future_position", "future_orientation_6d"):
        torch.testing.assert_close(adapted[name], expected[name])
    assert all(not parameter.requires_grad for parameter in adapter.base.parameters())
    assert all(parameter.requires_grad for parameter in adapter.calibration_parameters())


def test_streaming_function_returns_all_outputs_and_diagnostics(tmp_path):
    base = model()
    checkpoint = tmp_path / "model.pt"
    stats = {
        "emg": {"mean": np.zeros(8, dtype="float32"),
                "std": np.ones(8, dtype="float32")},
        "imu": {"mean": np.zeros(24, dtype="float32"),
                "std": np.ones(24, dtype="float32")},
        "position": {"mean": np.zeros(3, dtype="float32"),
                     "std": np.ones(3, dtype="float32")},
    }
    torch.save({"format": "gripper_neuromuscular_future_v1",
                "state_dict": base.state_dict(),
                "model_args": {"modality": "emg+imu", "width": 16,
                    "patch": 4, "stride": 2, "layers": 1, "heads": 4,
                    "dropout": 0., "react_context": 20,
                    "predict_click": True, "predict_final_pose": True,
                    "future_steps": 3},
                "normalization": stats,
                "preprocessing": {"raw_rate_hz": 1000., "rate_hz": 100.,
                                  "gap_s": .02}}, checkpoint)
    stream = NeuromuscularStream(checkpoint, "cpu", warmup_ms=20.)
    result = None
    for index in range(22):
        result = stream.update(index / 1000, np.zeros(4), np.zeros(24), (1440, 900))
    assert result is not None
    assert len(result["future_positions_m"]) == 3
    assert len(result["future_orientations_wxyz"]) == 3
    assert len(result["click_pixel_xy"]) == 2
    assert result["normalization_diagnostics"]["imu_fraction_abs_z_gt_10"] == 0


def test_calibration_training_smoke(tmp_path, monkeypatch):
    root = tmp_path / "data"; root.mkdir()
    for index in range(18):
        data = frame()
        data["gripper_state"] = np.where(
            data.time_perf_counter < 100.5, "open", "close")
        data["canvas_width_px"], data["canvas_height_px"] = 1440, 900
        data["click_x_norm"], data["click_y_norm"] = index / 18, .4
        data = data.drop(columns=[name for name in data if name.startswith("VIVE_")])
        data.to_csv(root / f"trial_{index:03}.csv", index=False)
    base = model()
    checkpoint, output = tmp_path / "base.pt", tmp_path / "calibrated.pt"
    stats = {"emg": {"mean": np.zeros(8), "std": np.ones(8)},
             "imu": {"mean": np.zeros(24), "std": np.ones(24)},
             "position": {"mean": np.zeros(3), "std": np.ones(3)}}
    torch.save({"format": "gripper_neuromuscular_future_v1",
                "state_dict": base.state_dict(),
                "model_args": {"modality": "emg+imu", "width": 16,
                    "patch": 4, "stride": 2, "layers": 1, "heads": 4,
                    "dropout": 0., "react_context": 20,
                    "predict_click": True, "predict_final_pose": True,
                    "future_steps": 3}, "normalization": stats,
                "preprocessing": {"raw_rate_hz": 1000., "rate_hz": 100.,
                                  "gap_s": .02, "event_origin": "auto",
                                  "event_pulse_s": .1, "require_events": False}}, checkpoint)
    monkeypatch.setattr(sys, "argv", ["calibrate", "--checkpoint", str(checkpoint),
        "--root", str(root), "--output", str(output), "--device", "cpu",
        "--epochs", "1", "--film-groups", "4"])
    calibration.main()
    saved = torch.load(output, weights_only=False)
    assert saved["format"] == "gripper_pixel_film_calibration_v1"
    assert saved["pose_and_future_frozen"] is True
    assert saved["vive_required"] is False
    assert saved["rejected"] == {}
