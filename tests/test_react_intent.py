import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

from emg_touch.models.reach_grasp_react_intent import CausalEMGIntent, ReactFutureIntentModel
from scripts.train_reach_grasp_react_intent import ReactTraining
from scripts import train_reach_grasp_future_intent as trainer
from test_reach_grasp import frame
from emg_touch.live_future_intent import LiveFutureIntentPredictor


def test_action_and_sensor_future_cannot_leak():
    torch.manual_seed(2)
    branch = CausalEMGIntent(width=16, layers=2, dropout=0.).eval()
    x = torch.randn(1, 24, 16)
    actions = torch.zeros(1, 24, dtype=torch.long)
    before = branch(x, actions)
    x[:, 15:] += 100
    actions[:, 15:] = 1
    after = branch(x, actions)
    for key in before:
        torch.testing.assert_close(before[key][:, :15], after[key][:, :15])
    masked = torch.full_like(actions, 2)
    torch.testing.assert_close(branch(x)["features"], branch(x, masked)["features"])


def test_chunked_attention_matches_full_receptive_field():
    branch = CausalEMGIntent(width=16, layers=2, dropout=0., context=10).eval()
    x = torch.randn(1, 43, 16)
    with torch.no_grad():
        full = branch(x)["features"]
        prefix = branch(x[:, :27])["features"]
    torch.testing.assert_close(full[:, :27], prefix)


def test_hidden_emg_values_cannot_be_copied():
    branch = CausalEMGIntent(width=16, layers=1, dropout=0.).eval()
    x = torch.randn(1, 20, 16)
    hidden = torch.zeros(1, 20, dtype=torch.bool)
    hidden[:, 5:10] = True
    first = branch(x, hidden=hidden)
    x[:, 5:10] += 100
    second = branch(x, hidden=hidden)
    torch.testing.assert_close(first["reconstruction"], second["reconstruction"])


def test_future_model_causality_loss_and_invalid_targets():
    torch.manual_seed(3)
    model = ReactFutureIntentModel(width=16, layers=1, patch=4, stride=2,
                                  heads=4, dropout=0.)
    x, imu = torch.randn(2, 24, 16), torch.randn(2, 24, 48)
    x[..., 8:] = 1
    imu[..., 24:] = 1
    output = model(x, imu)
    changed = x.clone()
    changed[:, 16:] += 30
    later = model(changed, imu)
    for key in ("future_position", "future_orientation_6d", "logits", "intent_time_logits"):
        torch.testing.assert_close(output[key][:, :16], later[key][:, :16])
    batch = {"emg": x, "labels": torch.zeros(2, 24, 3),
             "label_mask": torch.ones(2, 24, 3),
             "emg_usable": torch.ones(2, 24, dtype=torch.bool),
             "future_pose_mask": torch.ones(2, 24, 5, dtype=torch.bool),
             "pose_mask": torch.ones(2, 24, 1), "pose": torch.randn(2, 24, 3),
             "future_position": torch.randn(2, 24, 5, 3)}
    args = argparse.Namespace(masked_weight=.15, emg_holding_weight=.2,
                              stability_weight=.03, motion_weight=.2)
    loss, detail = ReactTraining.loss(model, output, batch, batch["emg_usable"], args)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.react.reconstruct.weight.grad.abs().sum() > 0
    assert model.motion_velocity.weight.grad.abs().sum() > 0
    batch["emg_usable"][:] = False
    batch["emg"][..., 8:] = 0
    batch["future_pose_mask"][:] = False
    output = model(x, imu)
    loss, _ = ReactTraining.loss(model, output, batch, batch["emg_usable"], args)
    assert loss.item() == 0


def test_training_and_live_checkpoint_smoke(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "data"
    root.mkdir()
    for index in range(20):
        data = pd.concat([frame(), frame()], ignore_index=True)
        data["time_perf_counter"] = 100 + np.arange(len(data)) / 1000
        data["t_grasp_perf"], data["t_release_perf"] = 100.5, 101.4
        data["EMG 1_S0"] += index * .01
        data["VIVE_T0_quat_w"] = 1.
        for axis in "xyz":
            data[f"VIVE_T0_quat_{axis}"] = 0.
        data.to_csv(root / f"trial_{index:03}.csv", index=False)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["train", "--root", str(root), "--output-dir", str(out),
        "--device", "cpu", "--epochs", "1", "--raw-rate-hz", "1000"])
    trainer.main(ReactFutureIntentModel, "reach_grasp_react_intent_v1", ReactTraining)
    result = json.loads((out / "results.json").read_text())
    assert np.isfinite(result["emg_imu"]["future"]["mean_future_position_cm"])
    assert (out / "stability.json").exists()
    runtime = LiveFutureIntentPredictor(out / "best.pt", "cpu", warmup_ms=50)
    for i in range(80):
        runtime.add_sample(i / 1000, np.ones(4) * .01, np.ones(24))
    prediction = runtime.predict()
    assert prediction["vive_is_model_input"] is False
    assert len(prediction["future_positions_m"]) == 5
