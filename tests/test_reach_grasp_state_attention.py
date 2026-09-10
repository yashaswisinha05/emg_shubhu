import json
import sys

import torch

from emg_touch.models.reach_grasp_state_attention import StateConditionedAttentionModel
from emg_touch.models.state_attention_loss import future_position_losses
from scripts import train_gripper_state_attention as training
from test_reach_grasp import frame


def inputs(length=24):
    emg, imu = torch.randn(2, length, 16), torch.randn(2, length, 48)
    emg[..., 8:], imu[..., 24:] = 1, 1
    return emg, imu


def model(modality="emg+imu"):
    return StateConditionedAttentionModel(
        modality=modality, width=16, patch=4, stride=2, layers=1,
        heads=4, dropout=0., future_steps=5).eval()


def test_outputs_are_causal_and_position_only():
    network = model()
    emg, imu = inputs()
    changed_emg, changed_imu = emg.clone(), imu.clone()
    changed_emg[:, 16:, :8] += 100
    changed_imu[:, 16:, :24] -= 100
    with torch.no_grad():
        first = network(emg, imu)
        changed = network(changed_emg, changed_imu)
    assert first["position"].shape == (2, 24, 3)
    assert first["click"].shape == (2, 24, 2)
    assert first["future_position"].shape == (2, 24, 5, 3)
    assert first["emg_channel_attention"].shape == (2, 24, 4)
    torch.testing.assert_close(first["position"][:, :16], changed["position"][:, :16])


def test_initial_emg_correction_is_exact_noop():
    network = model()
    assert network.correction[-1].weight.count_nonzero() == 0
    assert network.state_film.weight.count_nonzero() == 0


def test_state_is_emg_only():
    network = model()
    emg, imu = inputs()
    with torch.no_grad():
        first = network(emg, imu)["gripper_state_logits"]
        second = network(emg, imu + 100)["gripper_state_logits"]
    torch.testing.assert_close(first, second)


def test_future_consistency_has_gradient():
    output = {
        "position": torch.randn(1, 8, 3, requires_grad=True),
        "future_position": torch.randn(1, 8, 3, 3, requires_grad=True),
    }
    batch = {
        "emg_usable": torch.ones(1, 8, dtype=torch.bool),
        "imu_usable": torch.ones(1, 8, dtype=torch.bool),
        "pose_mask": torch.ones(1, 8, 1),
        "pose": torch.randn(1, 8, 3),
    }
    future, consistency = future_position_losses(output, batch)
    (future + consistency).backward()
    assert output["future_position"].grad.abs().sum() > 0
    # Arrival predictions are detached: consistency regularizes the future head.
    assert output["position"].grad is None


def test_training_entrypoint_smoke(tmp_path, monkeypatch):
    root, output = tmp_path / "data", tmp_path / "output"
    root.mkdir()
    for index in range(20):
        data = frame()
        data["gripper_state"] = "open" if index % 2 == 0 else "close"
        data["canvas_width_px"], data["canvas_height_px"] = 1440, 900
        data["click_x_norm"] = (index % 3 + 1) / 4
        data["click_y_norm"] = (index // 3 % 3 + 1) / 4
        data["EMG 1_S0"] += index * 1e-4
        data.to_csv(root / f"trial_{index:03}.csv", index=False)
    monkeypatch.setattr(sys, "argv", [
        "train", "--root", str(root), "--models", "emg+imu",
        "--device", "cpu", "--epochs", "1", "--batch-size", "4",
        "--raw-rate-hz", "1000", "--output-dir", str(output)])
    training.main()
    state = torch.load(output / "emg_imu_best.pt", weights_only=False)
    result = json.loads((output / "results.json").read_text())
    assert state["format"] == "gripper_state_attention_v1"
    assert result["protocol"]["orientation_and_endpoint_heads"] is False
