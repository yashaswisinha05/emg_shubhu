import json
import sys

import numpy as np
import torch

from emg_touch.data.wearable_augmentation import PhysiologicalWearableAugmenter
from emg_touch.models.reach_grasp_robust import RobustReachGraspModel
from emg_touch.live_reach_grasp import LiveReachGraspPredictor
from scripts import train_reach_grasp as base
from scripts import train_reach_grasp_robust as trainer
from tests.test_reach_grasp import frame
from tests.test_reach_grasp_orientation import add_orientation


def test_event_time_targets_include_soft_horizons_and_no_event():
    time = np.array([0., .1, .2, .3, .9])
    target = base.event_time_targets(time, np.array([.3, .8]), uncertainty_s=.04)
    assert target.shape == (5, 2, 6)
    np.testing.assert_allclose(target.sum(-1), 1.)
    assert target[0, 0, 3] > target[0, 0, 0]
    assert target[-1, 0, -1] == 1.


def test_robust_model_outputs_are_causal_and_well_shaped():
    torch.set_num_threads(1)
    model = RobustReachGraspModel(width=16, layers=1, heads=2,
                                  patch=8, stride=2).eval()
    emg, imu = torch.randn(2, 50, 16), torch.randn(2, 50, 48)
    before = model(emg, imu)
    assert before["logits"].shape == (2, 50, 3)
    assert before["event_time_logits"].shape == (2, 50, 2, 6)
    assert before["position_log_variance"].shape == (2, 50, 3)
    assert before["orientation_log_variance"].shape == (2, 50, 1)
    emg[:, 30:] += 1000
    imu[:, 30:] -= 1000
    after = model(emg, imu)
    for key in ["logits", "event_time_logits", "position",
                "orientation_6d", "position_log_variance"]:
        torch.testing.assert_close(before[key][:, :30], after[key][:, :30],
                                   atol=1e-5, rtol=1e-5)


def test_physiological_augmentation_preserves_packed_contract():
    torch.manual_seed(4)
    batch = {
        "emg": torch.cat([torch.zeros(3, 40, 8), torch.ones(3, 40, 8)], -1),
        "imu": torch.cat([torch.zeros(3, 40, 24), torch.ones(3, 40, 24)], -1),
    }
    stats = {
        "emg": {"mean": [1.] * 8, "std": [.2] * 8},
        "imu": {"mean": [0.] * 24, "std": [1.] * 24},
    }
    result = PhysiologicalWearableAugmenter(1.)(batch, stats, "emg+imu")
    assert result["emg"].shape == (3, 40, 16)
    assert result["imu"].shape == (3, 40, 48)
    assert torch.isfinite(result["emg"]).all()
    assert torch.isfinite(result["imu"]).all()
    assert result["emg_usable"].shape == (3, 40)


def test_robust_training_smoke(tmp_path, monkeypatch):
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
    results = json.loads((output / "results.json").read_text())
    assert state["format"] == "reach_grasp_robust_v1"
    assert state["robust_training"]["physiological_augmentation"]
    assert results["emg+imu"]["event_time_mae_ms_within_450ms"] >= 0
    assert results["emg+imu"]["position_predicted_1sigma_cm"] >= 0
    assert results["emg+imu"]["orientation_predicted_1sigma_deg"] >= 0
    live = LiveReachGraspPredictor(output / "emg_imu_best.pt", "cpu")
    assert isinstance(live.model, RobustReachGraspModel)
