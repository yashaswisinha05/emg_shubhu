import pytest
import torch
import json
import sys
import numpy as np

from emg_touch.models.reach_grasp_architecture_baselines import ArchitectureBaseline
from scripts import train_architecture_baseline as trainer
from test_reach_grasp import frame


@pytest.mark.parametrize("architecture", sorted(ArchitectureBaseline.NAMES))
def test_comparison_baseline_shapes_and_causality(architecture):
    initialization = {
        "state_logits": [0., 1.], "position": [1., 2., 3.],
        "click": [.25, .75], "future_position": [[0., 0., 0.]] * 3,
    }
    model = ArchitectureBaseline(
        architecture, width=16, patch=4, stride=2, layers=1, heads=4,
        dropout=0., future_steps=3, constant_initialization=initialization).eval()
    emg, imu = torch.randn(2, 24, 16), torch.randn(2, 24, 48)
    with torch.no_grad():
        original = model(emg, imu)
        changed_emg, changed_imu = emg.clone(), imu.clone()
        changed_emg[:, 13:] += 100
        changed_imu[:, 13:] -= 100
        changed = model(changed_emg, changed_imu)
    assert original["gripper_state_logits"].shape == (2, 24, 2)
    assert original["position"].shape == (2, 24, 3)
    assert original["click"].shape == (2, 24, 2)
    assert original["future_position"].shape == (2, 24, 3, 3)
    torch.testing.assert_close(original["position"][:, :13],
                               changed["position"][:, :13], atol=2e-5, rtol=2e-5)


def test_recurrent_models_are_not_bidirectional():
    for architecture in ("gru", "lstm"):
        model = ArchitectureBaseline(architecture, width=16, layers=2,
                                     heads=4, future_steps=3)
        assert model.recurrent.bidirectional is False


def test_constant_uses_supplied_training_statistics():
    initialization = {
        "state_logits": [-2., 2.], "position": [1., 2., 3.],
        "click": [.2, .8], "future_position": [[4., 5., 6.]] * 2,
    }
    model = ArchitectureBaseline(
        "constant", width=16, future_steps=2,
        constant_initialization=initialization)
    output = model(torch.randn(1, 5, 16), torch.randn(1, 5, 48))
    torch.testing.assert_close(output["position"][0, 0], torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(output["click"][0, 0], torch.tensor([.2, .8]))


def test_lstm_comparison_training_entrypoint(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "data"
    root.mkdir()
    for index in range(20):
        data = frame()
        data["EMG 1_S0"] += index * .01
        data["gripper_state"] = np.where(
            data.time_perf_counter < 100.5, "open", "close")
        data["canvas_width_px"], data["canvas_height_px"] = 1920, 1080
        data["click_x_pos"] = 200 + (index % 5) * 350
        data["click_y_pos"] = 150 + (index % 4) * 240
        data.to_csv(root / f"trial_{index:03}.csv", index=False)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "train", "--architecture", "lstm", "--root", str(root),
        "--output-dir", str(output), "--device", "cpu", "--epochs", "1",
        "--batch-size", "4", "--raw-rate-hz", "1000",
        "--final-pose-weight", "0", "--models", "emg+imu"])
    trainer.main()
    result = json.loads((output / "results.json").read_text())
    assert result["protocol"]["architecture"] == "lstm"
    checkpoint = torch.load(output / "emg_imu_best.pt", weights_only=False)
    assert checkpoint["format"] == "reach_grasp_architecture_baseline_v1"
    assert checkpoint["architecture"] == "lstm"
    assert checkpoint["parameter_count"] > 0
