import json
import sys

import numpy as np
import torch

from emg_touch.data.reach_grasp import preprocess
from emg_touch.models.reach_grasp_orientation_hybrid import ReachGraspOrientationHybrid
from emg_touch.physics.rotation_6d import (
    matrix_to_rotation_6d_numpy, orientation_errors_numpy,
    quaternion_to_matrix_numpy)
from scripts import train_reach_grasp_hybrid as hybrid_trainer
from scripts import train_reach_grasp_orientation as orientation_trainer
from tests.test_reach_grasp import frame, settings


def add_orientation(data, yaw_degrees=0.):
    angle = np.radians(yaw_degrees) / 2
    data["VIVE_T0_quat_w"] = np.cos(angle)
    data["VIVE_T0_quat_x"] = 0.
    data["VIVE_T0_quat_y"] = 0.
    data["VIVE_T0_quat_z"] = np.sin(angle)
    return data


def test_rotation_representation_and_circular_yaw():
    quaternion = np.array([[1., 0, 0, 0], [np.sqrt(.5), 0, 0, np.sqrt(.5)]])
    rotation = quaternion_to_matrix_numpy(quaternion)
    six = matrix_to_rotation_6d_numpy(rotation)
    geodesic, yaw = orientation_errors_numpy(six, six)
    np.testing.assert_allclose(geodesic, 0, atol=1e-5)
    np.testing.assert_allclose(yaw, 0, atol=1e-5)
    identity = np.repeat(matrix_to_rotation_6d_numpy(np.eye(3))[None], 2, axis=0)
    geodesic, yaw = orientation_errors_numpy(identity, six)
    assert np.isclose(geodesic[1], 90., atol=1e-4)
    assert np.isclose(yaw[1], 90., atol=1e-4)


def test_orientation_preprocessing_and_model_causality(tmp_path):
    path = tmp_path / "trial.csv"
    add_orientation(frame(), 30).to_csv(path, index=False)
    trial = preprocess(path, settings())
    assert trial["orientation"].shape == (100, 6)
    assert trial["orientation_valid"].all()
    model = ReachGraspOrientationHybrid(width=16, layers=1, heads=2,
                                        patch=8, stride=2).eval()
    emg, imu = torch.randn(1, 40, 16), torch.randn(1, 40, 48)
    before = model(emg, imu)
    assert before["orientation_6d"].shape == (1, 40, 6)
    emg[:, 25:] += 1000
    imu[:, 25:] -= 1000
    after = model(emg, imu)
    torch.testing.assert_close(before["orientation_6d"][:, :25],
                               after["orientation_6d"][:, :25], atol=1e-5, rtol=1e-5)


def test_orientation_training_smoke(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "data"
    root.mkdir()
    for index in range(20):
        data = add_orientation(frame(), index)
        data["EMG 1_S0"] += index * .01
        data.to_csv(root / f"trial_{index:03}.csv", index=False)
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["train", "--root", str(root),
        "--output-dir", str(output), "--device", "cpu", "--epochs", "1",
        "--raw-rate-hz", "1000", "--models", "emg", "--batch-size", "4"])
    monkeypatch.setattr(hybrid_trainer, "MODEL_EXTRA_ARGS",
        {"width": 16, "layers": 1, "heads": 2, "patch": 8, "stride": 2})
    orientation_trainer.main()
    state = torch.load(output / "emg_best.pt", weights_only=False)
    results = json.loads((output / "results.json").read_text())
    assert state["format"] == "reach_grasp_orientation_hybrid_v1"
    assert state["orientation_representation"].startswith("6D")
    assert results["emg"]["orientation_geodesic_deg"] >= 0
    assert results["hybrid_event_decoding"]["emg"]["combined"]["yaw_mae_deg"] >= 0
