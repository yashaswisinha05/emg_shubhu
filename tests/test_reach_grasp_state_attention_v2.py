import json
import sys

import torch

from emg_touch.models.reach_grasp_state_attention_v2 import StateConditionedAttentionV2
from scripts import train_gripper_state_attention_v2 as training
from test_reach_grasp import frame


def model():
    return StateConditionedAttentionV2(
        width=16, patch=4, stride=2, layers=1, heads=4,
        dropout=0., future_steps=5).eval()


def inputs():
    emg, imu = torch.randn(2, 24, 16), torch.randn(2, 24, 48)
    emg[..., 8:], imu[..., 24:] = 1, 1
    return emg, imu


def test_v2_outputs_strong_state_and_geometric_pixel_paths():
    output = model()(*inputs())
    assert output["gripper_state_logits"].shape == (2, 24, 2)
    assert output["react_reconstruction"].shape == (2, 24, 8)
    assert output["click_direct"].shape == (2, 24, 2)
    assert output["click_from_position"].shape == (2, 24, 2)
    assert output["click_position_blend"].shape == (2, 24, 1)


def test_v2_is_causal_and_state_is_imu_invariant():
    network = model()
    emg, imu = inputs()
    changed_emg, changed_imu = emg.clone(), imu.clone()
    changed_emg[:, 16:, :8] += 100
    changed_imu[:, 16:, :24] -= 100
    with torch.no_grad():
        first = network(emg, imu)
        changed = network(changed_emg, changed_imu)
        imu_changed_state = network(emg, imu + 100)["gripper_state_logits"]
    torch.testing.assert_close(first["position"][:, :16], changed["position"][:, :16])
    torch.testing.assert_close(first["gripper_state_logits"], imu_changed_state)


def test_pixel_loss_cannot_bend_position_head():
    network = model()
    output = network(*inputs())
    output["click_from_position"].sum().backward()
    assert all(parameter.grad is None for parameter in network.position.parameters())


def test_v2_training_entrypoint_smoke(tmp_path, monkeypatch):
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
        "train", "--root", str(root), "--models", "emg+imu", "--device", "cpu",
        "--epochs", "1", "--batch-size", "4", "--raw-rate-hz", "1000",
        "--output-dir", str(output)])
    training.main()
    state = torch.load(output / "emg_imu_best.pt", weights_only=False)
    result = json.loads((output / "results.json").read_text())
    assert state["format"] == "gripper_state_attention_v2"
    assert result["protocol"]["architecture"] == "state-conditioned-attention-v2"
