import sys

import numpy as np
import torch

from emg_touch.models.hard_contrastive import hard_trial_contrastive_loss
from emg_touch.models.pixel_gripper_film import GripperHardContrastiveFiLM
from emg_touch.models.reach_grasp_neuromuscular_future import (
    NeuromuscularFutureGripperPoseModel,
)
from emg_touch.neuromuscular_inference import load_neuromuscular_model
from scripts import calibrate_gripper_hard_contrastive as calibration
from test_reach_grasp import frame


def small_model():
    return NeuromuscularFutureGripperPoseModel(
        modality="emg+imu", width=16, patch=4, stride=2, layers=1,
        heads=4, dropout=0., react_context=20, predict_click=True,
        predict_final_pose=True, future_steps=3).eval()


def checkpoint(path, model):
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


def test_hard_contrastive_rewards_class_separation():
    labels = torch.tensor([[0], [0], [1], [1]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    separated = torch.tensor([[[1., 0.]], [[.9, .1]],
                              [[-1., 0.]], [[-.9, -.1]]])
    collapsed = torch.tensor([[[1., 0.]], [[.9, .1]],
                              [[1., 0.]], [[.9, .1]]])
    assert hard_trial_contrastive_loss(separated, labels, valid) < (
        hard_trial_contrastive_loss(collapsed, labels, valid))


def test_gripper_adapter_identity_preserves_every_output_except_features():
    base = small_model()
    adapter = GripperHardContrastiveFiLM(base, groups=4, rank=4).eval()
    emg, imu = torch.randn(1, 20, 16), torch.randn(1, 20, 48)
    emg[..., 8:], imu[..., 24:] = 1, 1
    with torch.no_grad():
        expected, actual = base(emg, imu), adapter(emg, imu)
    for name in ("click", "gripper_state_logits", "position", "orientation_6d",
                 "final_position", "future_position"):
        torch.testing.assert_close(actual[name], expected[name])


def test_20_plus_20_training_without_vive(tmp_path, monkeypatch):
    open_root, close_root = tmp_path / "open", tmp_path / "close"
    open_root.mkdir(); close_root.mkdir()
    for label, root in (("open", open_root), ("close", close_root)):
        for index in range(20):
            data = frame()
            data["gripper_state"] = label
            data = data.drop(columns=[name for name in data if name.startswith("VIVE_")])
            data.to_csv(root / f"trial_{index:03}.csv", index=False)
    base_path, output = tmp_path / "base.pt", tmp_path / "contrastive.pt"
    checkpoint(base_path, small_model())
    monkeypatch.setattr(sys, "argv", ["calibrate", "--checkpoint", str(base_path),
        "--open-root", str(open_root), "--close-root", str(close_root),
        "--output", str(output), "--device", "cpu", "--epochs", "1",
        "--film-groups", "4", "--adapter-rank", "4"])
    calibration.main()
    saved = torch.load(output, weights_only=False)
    assert saved["format"] == "gripper_hard_contrastive_calibration_v1"
    assert saved["trials_per_class"] == 20
    loaded, _ = load_neuromuscular_model(output, "cpu")
    assert isinstance(loaded, GripperHardContrastiveFiLM)
