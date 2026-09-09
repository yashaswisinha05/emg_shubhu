import torch
import json
import sys

import numpy as np
import pandas as pd

from emg_touch.data.gripper_state import add_gripper_state
from emg_touch.data.reach_grasp import preprocess
from emg_touch.models.reach_grasp_gripper_state import GripperStatePoseModel
from scripts import train_gripper_state_pose as trainer
from test_reach_grasp import frame, settings


def test_shapes_and_causality():
    torch.manual_seed(5)
    model = GripperStatePoseModel(
        width=16, patch=4, stride=2, layers=1, heads=4, dropout=0.).eval()
    emg, imu = torch.randn(2, 30, 16), torch.randn(2, 30, 48)
    emg[..., 8:] = 1
    imu[..., 24:] = 1
    changed_emg, changed_imu = emg.clone(), imu.clone()
    changed_emg[:, 20:, :8] += 50
    changed_imu[:, 20:, :24] += 50
    with torch.no_grad():
        output = model(emg, imu)
        changed = model(changed_emg, changed_imu)

    assert output["position"].shape == (2, 30, 3)
    assert output["orientation_6d"].shape == (2, 30, 6)
    assert output["gripper_state_logits"].shape == (2, 30, 2)
    assert output["react_gripper_state_logits"].shape == (2, 30, 2)

    # No future sample may move an earlier prediction.
    for key in ("position", "orientation_6d", "gripper_state_logits"):
        torch.testing.assert_close(output[key][:, :20], changed[key][:, :20])


def test_react_correction_starts_as_a_no_op():
    """react_current is zero-initialized: with a freshly constructed model,
    disabling the react correction (by zeroing its output) must not change
    gripper_state_logits at all."""
    torch.manual_seed(6)
    model = GripperStatePoseModel(
        width=16, patch=4, stride=2, layers=1, heads=4, dropout=0.).eval()
    assert torch.count_nonzero(model.react_current.weight) == 0
    assert torch.count_nonzero(model.react_current.bias) == 0
    emg, imu = torch.randn(1, 12, 16), torch.randn(1, 12, 48)
    emg[..., 8:] = 1
    imu[..., 24:] = 1
    with torch.no_grad():
        output = model(emg, imu)
        # branch["features"] is nonzero (random init), but react_current maps
        # anything to exactly zero right after construction.
        branch = model.react(emg)
        correction = model.react_current(branch["features"])
    assert torch.count_nonzero(branch["features"]) > 0
    torch.testing.assert_close(correction, torch.zeros_like(correction))
    assert torch.isfinite(output["gripper_state_logits"]).all()


def test_modality_ablation_zeros_the_other_branch():
    torch.manual_seed(7)
    emg_only = GripperStatePoseModel(
        modality="emg", width=16, patch=4, stride=2, layers=1, heads=4, dropout=0.).eval()
    emg, imu = torch.randn(1, 16, 16), torch.randn(1, 16, 48)
    emg[..., 8:] = 1
    imu[..., 24:] = 1
    with torch.no_grad():
        baseline = emg_only(emg, imu)
        changed = emg_only(emg, imu + 100)
    torch.testing.assert_close(baseline["position"], changed["position"])
    torch.testing.assert_close(baseline["fusion_weights"][..., 1],
                               torch.zeros_like(baseline["fusion_weights"][..., 1]))

    imu_only = GripperStatePoseModel(
        modality="imu", width=16, patch=4, stride=2, layers=1, heads=4, dropout=0.).eval()
    with torch.no_grad():
        baseline = imu_only(emg, imu)
        changed = imu_only(emg + 100, imu)
    torch.testing.assert_close(baseline["gripper_state_logits"],
                               changed["gripper_state_logits"])


def test_gripper_state_alignment_and_unknown_label(tmp_path):
    data = frame()
    data = data.drop(columns=["t_grasp_perf", "t_release_perf"])
    data["gripper_state"] = np.where(data.time_perf_counter < 100.3, "open",
                              np.where(data.time_perf_counter < 100.7, "close", "open"))
    data.loc[305:307, "gripper_state"] = None
    path = tmp_path / "trial.csv"
    data.to_csv(path, index=False)
    state_settings = {**settings(), "require_events": False}
    trial = add_gripper_state(path, preprocess(path, state_settings))
    assert trial["gripper_state"][10] == 0
    assert trial["gripper_state"][40] == 1
    assert trial["gripper_state"][80] == 0
    data.loc[10, "gripper_state"] = "ajar"
    data.to_csv(path, index=False)
    try:
        add_gripper_state(path, preprocess(path, state_settings))
        assert False, "unknown labels must be rejected"
    except ValueError as error:
        assert "unknown gripper_state" in str(error)


def test_training_smoke(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "data"
    root.mkdir()
    for index in range(20):
        data = frame()
        data["EMG 1_S0"] += index * .01
        data["gripper_state"] = np.where(data.time_perf_counter < 100.3, "open",
                                  np.where(data.time_perf_counter < 100.7, "close", "open"))
        data.to_csv(root / f"trial_{index:03}.csv", index=False)
    real = GripperStatePoseModel
    monkeypatch.setattr(trainer, "GripperStatePoseModel", lambda **kwargs: real(
        modality=kwargs.get("modality", "emg+imu"),
        width=16, patch=4, stride=2, layers=1, heads=4, dropout=0.))
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train", "--root", str(root), "--output-dir",
        str(output), "--device", "cpu", "--epochs", "1", "--raw-rate-hz", "1000"])
    trainer.main()
    result = json.loads((output / "results.json").read_text())
    # Default --models trains all three: a real ablation (dedicated
    # emg-only/imu-only models), not just the fused model probed with one
    # input zeroed out.
    assert result["protocol"]["models"] == ["emg", "imu", "emg+imu"]
    for modality in ("emg", "imu", "emg+imu"):
        assert set(result[modality]) >= {"gripper_macro_f1", "position_cm"}
        checkpoint = torch.load(output / f"{modality.replace('+', '_')}_best.pt",
                                weights_only=False)
        assert checkpoint["classes"] == ["open", "close"]
        assert checkpoint["model_args"]["modality"] == modality
    assert set(result["fusion_zero_emg"]) >= {"gripper_macro_f1", "position_cm"}
    assert set(result["fusion_zero_imu"]) >= {"gripper_macro_f1", "position_cm"}
