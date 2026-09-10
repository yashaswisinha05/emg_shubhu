import json
import sys

import numpy as np
import torch

from emg_touch.models.reach_grasp_timesiam_future import TimeSiamFutureGripperPoseModel
from emg_touch.models.timesiam_wearable_loss import future_wearable_loss
from scripts import train_gripper_timesiam_future as trainer
from test_reach_grasp import frame


def inputs(length=24):
    emg, imu = torch.randn(1, length, 16), torch.randn(1, length, 48)
    emg[..., 8:], imu[..., 24:] = 1, 1
    return emg, imu


def test_timesiam_future_is_causal_and_uses_lineages():
    torch.manual_seed(4)
    model = TimeSiamFutureGripperPoseModel(
        width=16, patch=4, stride=2, layers=1, heads=4, dropout=0.,
        predict_click=True, predict_final_pose=True, future_steps=5,
        lineage_context=4).eval()
    emg, imu = inputs()
    changed_emg, changed_imu = emg.clone(), imu.clone()
    changed_emg[:, 16:, :8] += 100
    changed_imu[:, 16:, :24] -= 100
    with torch.no_grad():
        output = model(emg, imu)
        changed = model(changed_emg, changed_imu)
    assert output["future_position"].shape == (1, 24, 5, 3)
    assert output["future_wearable"].shape == (1, 24, 5, 32)
    torch.testing.assert_close(
        output["future_position"][:, :16], changed["future_position"][:, :16])
    assert not torch.allclose(output["lineage_features"][:, :, 0],
                              output["lineage_features"][:, :, 1])


def test_pretraining_loss_respects_modality_and_future_alignment():
    output = {"future_wearable": torch.zeros(1, 5, 2, 32, requires_grad=True)}
    emg = torch.zeros(1, 5, 16); imu = torch.zeros(1, 5, 48)
    emg[..., 8:], imu[..., 24:] = 1, 1
    emg[:, 1:, :8] = 2
    imu[:, 1:, :24] = 50
    batch = {"emg": emg, "imu": imu,
             "emg_usable": torch.ones(1, 5, dtype=torch.bool),
             "imu_usable": torch.ones(1, 5, dtype=torch.bool)}
    emg_loss = future_wearable_loss(output, batch, "emg")
    both_loss = future_wearable_loss(output, batch, "emg+imu")
    assert both_loss > emg_loss
    emg_loss.backward()
    assert output["future_wearable"].grad[..., :8].abs().sum() > 0
    assert output["future_wearable"].grad[..., 8:].abs().sum() == 0


def test_timesiam_training_smoke(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "data"; root.mkdir()
    for index in range(20):
        data = frame()
        data["EMG 1_S0"] += index * .01
        data["gripper_state"] = np.where(data.time_perf_counter < 100.5, "open", "close")
        data["canvas_width_px"], data["canvas_height_px"] = 1440, 900
        data["click_x_norm"], data["click_y_norm"] = index / 20, .4
        data.to_csv(root / f"trial_{index:03}.csv", index=False)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train", "--root", str(root),
        "--output-dir", str(output), "--device", "cpu", "--models", "emg+imu",
        "--epochs", "1", "--raw-rate-hz", "1000", "--timesiam-pretrain-epochs", "1",
        "--timesiam-width", "16", "--timesiam-patch", "4", "--timesiam-stride", "2",
        "--timesiam-layers", "1", "--timesiam-heads", "4",
        "--timesiam-lineage-context", "4"])
    trainer.main()
    result = json.loads((output / "results.json").read_text())
    assert result["protocol"]["future_architecture"].startswith("TimeSiam")
    checkpoint = torch.load(output / "emg_imu_best.pt", weights_only=False)
    assert checkpoint["format"] == "gripper_timesiam_future_v1"
    assert checkpoint["model_args"]["future_steps"] == 20
    assert (output / "emg_imu_timesiam_pretrained.pt").exists()
