import pytest
import torch
import json
import sys
import numpy as np

from emg_touch.models.reach_grasp_architecture_baselines import ArchitectureBaseline
from emg_touch.models.reach_grasp_residual_gru import NeuromuscularResidualGRU
from scripts import train_architecture_baseline as trainer
from scripts import train_neuromuscular_residual_gru as residual_trainer
from scripts import run_gripper_pose_architecture_study as study
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


def test_architecture_study_resume_skips_only_complete_runs(tmp_path, monkeypatch):
    complete = tmp_path / "complete"
    complete.mkdir()
    for name in study.REQUIRED_RUN_FILES:
        (complete / name).write_text("done")
    calls = []
    monkeypatch.setattr(study, "run", lambda command, dry_run: calls.append(command))
    study.run_or_resume(["train"], complete, resume=True, dry_run=False)
    assert calls == []

    missing = tmp_path / "missing"
    study.run_or_resume(["train"], missing, resume=True, dry_run=False)
    assert calls == [["train"]]


def test_architecture_study_resume_rejects_partial_run(tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "results.json").write_text("{}")
    with pytest.raises(RuntimeError, match="cannot resume incomplete run"):
        study.run_or_resume(["train"], partial, resume=True, dry_run=False)


def test_residual_gru_heads_are_causal_and_state_is_emg_only():
    model = NeuromuscularResidualGRU(
        width=16, layers=1, dropout=0., future_steps=3).eval()
    emg, imu = torch.randn(2, 24, 16), torch.randn(2, 24, 48)
    with torch.no_grad():
        original = model(emg, imu)
        changed_emg, changed_imu = emg.clone(), imu.clone()
        changed_emg[:, 13:] += 100
        changed_imu[:, 13:] -= 100
        changed = model(changed_emg, changed_imu)
        imu_changed = model(emg, imu + 100)
    assert original["position"].shape == (2, 24, 3)
    assert original["click"].shape == (2, 24, 2)
    assert original["grid_logits"].shape == (2, 24, 9)
    assert original["future_position"].shape == (2, 24, 3, 3)
    assert original["future_imu_delta"].shape == (2, 24, 3, 24)
    assert original["emg_reconstruction"].shape == (2, 24, 8)
    torch.testing.assert_close(original["position"][:, :13],
                               changed["position"][:, :13])
    torch.testing.assert_close(original["gripper_state_logits"],
                               imu_changed["gripper_state_logits"])
    assert bool(((original["click"] >= 0) & (original["click"] <= 1)).all())


def test_residual_gru_emg_motion_branch_starts_as_no_op():
    model = NeuromuscularResidualGRU(width=16, layers=1, future_steps=2).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 8, 16), torch.randn(1, 8, 48))
    torch.testing.assert_close(output["emg_correction"],
                               torch.zeros_like(output["emg_correction"]))


def test_residual_gru_training_entrypoint(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "data"
    root.mkdir()
    for index in range(20):
        data = frame()
        data["EMG 1_S0"] += index * .01
        data["gripper_state"] = np.where(
            data.time_perf_counter < 100.5, "open", "close")
        data["canvas_width_px"], data["canvas_height_px"] = 1920, 1080
        data["click_x_pos"] = 200 + (index % 3) * 700
        data["click_y_pos"] = 150 + (index % 3) * 350
        data.to_csv(root / f"trial_{index:03}.csv", index=False)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "train", "--root", str(root), "--output-dir", str(output),
        "--device", "cpu", "--epochs", "1", "--batch-size", "4",
        "--raw-rate-hz", "1000", "--future-pose-ms", "20",
        "--final-pose-weight", "0", "--models", "emg+imu"])
    residual_trainer.main()
    checkpoint = torch.load(output / "emg_imu_best.pt", weights_only=False)
    assert checkpoint["format"] == "neuromuscular_residual_gru_v1"
    assert checkpoint["parameter_count"] > 0
    assert checkpoint["model_args"]["future_steps"] == 2
