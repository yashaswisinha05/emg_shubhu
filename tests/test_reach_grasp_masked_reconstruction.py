import json
import sys

import numpy as np
import torch

from emg_touch.live_reach_grasp import LiveReachGraspPredictor
from emg_touch.models.reach_grasp_masked_reconstruction import (
    MaskedReconstructionReachGraspModel)
from scripts import train_reach_grasp as base
from scripts import train_reach_grasp_masked_reconstruction as trainer
from tests.test_reach_grasp import frame
from tests.test_reach_grasp_orientation import add_orientation


def test_masking_hides_only_observed_emg_and_preserves_targets():
    torch.manual_seed(2)
    values = torch.arange(2 * 12 * 8, dtype=torch.float32).reshape(2, 12, 8)
    valid = torch.ones_like(values)
    valid[:, 3, 2] = 0
    packed = torch.cat([values, valid], -1)
    masked, target, hidden = base.masked_emg_input(packed, ratio=.5, span=3)
    torch.testing.assert_close(target, values)
    assert hidden.any()
    assert not hidden[:, 3, 2].any()
    assert torch.all(masked[..., :8][hidden] == 0)
    assert torch.all(masked[..., 8:][hidden] == 0)


def test_reconstruction_head_is_causal_and_uses_emg_context():
    torch.set_num_threads(1)
    model = MaskedReconstructionReachGraspModel(
        width=16, layers=1, heads=2, patch=8, stride=2).eval()
    emg, imu = torch.randn(2, 40, 16), torch.randn(2, 40, 48)
    before = model(emg, imu)["emg_reconstruction"]
    emg[:, 25:] += 1000
    imu[:, 25:] -= 1000
    after = model(emg, imu)["emg_reconstruction"]
    assert before.shape == (2, 40, 8)
    torch.testing.assert_close(before[:, :25], after[:, :25], atol=1e-5, rtol=1e-5)


def test_masked_reconstruction_training_and_live_loading(tmp_path, monkeypatch):
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
        "--raw-rate-hz", "1000", "--models", "emg+imu", "--batch-size", "4"])
    monkeypatch.setattr(trainer, "MODEL_EXTRA_ARGS",
        {"width": 16, "layers": 1, "heads": 2, "patch": 8,
         "stride": 2, "event_time_bins": 6})
    trainer.main()
    state = torch.load(output / "emg_imu_best.pt", weights_only=False)
    history = json.loads((output / "emg_imu_history.json").read_text())
    assert state["format"] == "reach_grasp_masked_reconstruction_v1"
    assert state["robust_training"]["masked_emg_reconstruction_weight"] == .1
    assert history[0]["masked_emg_reconstruction_mse"] >= 0
    live = LiveReachGraspPredictor(output / "emg_imu_best.pt", "cpu")
    assert isinstance(live.model, MaskedReconstructionReachGraspModel)
