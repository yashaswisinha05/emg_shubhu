import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from emg_touch.data.reach_grasp import preprocess, event_times, past_fill, SENSORS
from emg_touch.models.reach_grasp import ReachGraspModel
from scripts import train_reach_grasp as trainer


def frame():
    n = 1000
    rng = np.random.default_rng(2)
    f = pd.DataFrame({"time_perf_counter": 100 + np.arange(n) / 1000,
                      "t_grasp_perf": 100.3, "t_release_perf": 100.7,
                      "sample_rate_hz_declared": 1000.})
    for s in SENSORS:
        f[f"EMG 1_{s}"] = rng.normal(size=n)
        for kind in ["ACC", "GYRO"]:
            for axis in "XYZ":
                f[f"{kind} {axis}_{s}"] = rng.normal(size=n)
    for axis in "xyz":
        f[f"VIVE_T0_pos_{axis}_m"] = np.arange(n) / 10000
    return f


def settings():
    return {"event_origin": "auto", "raw_rate_hz": 1000., "rate_hz": 100.,
            "gap_s": .02, "event_pulse_s": .1}


def test_metadata_and_gaps():
    f = frame()
    assert event_times(f)[1] == "absolute_perf"
    f.loc[1, "t_grasp_perf"] = 90
    with pytest.raises(ValueError, match="inconsistent"):
        event_times(f)
    with pytest.raises(ValueError, match="confirm"):
        event_times(pd.DataFrame({"grasp_onset_s": [.3], "grasp_offset_s": [.7]}))
    values, valid = past_fill(np.array([0., .01, .1]), np.array([[2.], [np.nan], [np.nan]]), .02)
    assert values[:, 0].tolist() == [2., 2., 0.]
    assert valid[:, 0].tolist() == [True, True, False]


def test_preprocessing_causality_and_masks(tmp_path):
    f = frame()
    f.loc[300:330, "VIVE_T0_pos_x_m"] = np.nan
    path = tmp_path / "trial.csv"
    f.to_csv(path, index=False)
    first = preprocess(path, settings())
    f.loc[700:, "EMG 1_S0"] = 1e6
    f.loc[700:, "ACC X_S0"] = -1e6
    f.to_csv(path, index=False)
    second = preprocess(path, settings())
    np.testing.assert_allclose(first["emg"][:65], second["emg"][:65])
    np.testing.assert_allclose(first["imu"][:65], second["imu"][:65])
    assert not first["pose_valid"][31]
    assert first["emg_valid"][31].all()


def test_model_causality_and_event_matching():
    torch.set_num_threads(1)
    model = ReachGraspModel().eval()
    emg, imu = torch.randn(2, 80, 16), torch.randn(2, 80, 48)
    first = model(emg, imu)
    emg[:, 50:] += 1000
    second = model(emg, imu)
    torch.testing.assert_close(first["logits"][:, :50], second["logits"][:, :50])
    item = {"trial": {"events": np.array([.3, .7]), "time": np.arange(100) / 100},
            "prob": np.zeros((100, 3))}
    item["prob"][30:35, 1] = 1
    report = trainer.event_summary([item], 0, .5, .15)
    assert report["f1"] == 1 and report["matched_mae_ms"] == 0


def test_full_training_smoke(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "data"
    root.mkdir()
    for i in range(20):
        f = frame()
        f["EMG 1_S0"] += i * .01  # Distinct files, not byte-duplicate trials.
        f.to_csv(root / f"trial_{i:03}.csv", index=False)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["train", "--root", str(root), "--output-dir", str(out),
                        "--device", "cpu", "--epochs", "1", "--raw-rate-hz", "1000"])
    trainer.main()
    results = json.loads((out / "results.json").read_text())
    assert {"emg", "imu", "emg+imu", "fusion_zero_emg", "schedule_only_baseline"} == set(results)
    state = torch.load(out / "emg_imu_best.pt", weights_only=False)
    assert state["format"] == "reach_grasp_v1"
    model = ReachGraspModel(**state["model_args"])
    model.load_state_dict(state["state_dict"])
    assert set(state["normalization"]) == {"emg", "imu", "position"}
