import numpy as np
import torch
import json
import sys

from emg_touch.data.hybrid_event_decoder import apply_decoder, calibrate_decoder
from emg_touch.models.reach_grasp_hybrid import ReachGraspHybrid
from scripts import train_reach_grasp_hybrid as trainer
from tests.test_reach_grasp import frame


def test_hybrid_shape_causality_and_local_head():
    torch.set_num_threads(1)
    model = ReachGraspHybrid(width=16, layers=1, heads=2, patch=8, stride=2).eval()
    emg, imu = torch.randn(2, 50, 16), torch.randn(2, 50, 48)
    before = model(emg, imu)
    assert before["logits"].shape == (2, 50, 3)
    assert before["position"].shape == (2, 50, 3)
    assert before["local_fusion_weights"].shape == (2, 50, 2)
    emg[:, 30:] += 1000
    imu[:, 30:] -= 1000
    after = model(emg, imu)
    torch.testing.assert_close(before["logits"][:, :30], after["logits"][:, :30],
                               atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(before["position"][:, :30], after["position"][:, :30],
                               atol=1e-5, rtol=1e-5)


def test_unimodal_hybrid_ignores_other_branch():
    torch.set_num_threads(1)
    model = ReachGraspHybrid("imu", width=16, layers=1, heads=2,
                             patch=8, stride=2).eval()
    emg, imu = torch.randn(1, 30, 16), torch.randn(1, 30, 48)
    first = model(emg, imu)
    second = model(emg + 10000, imu)
    torch.testing.assert_close(first["logits"], second["logits"])
    torch.testing.assert_close(first["position"], second["position"])


def item(local_grasp=True):
    time = np.arange(0, 1., .01)
    holding = np.full(len(time), .05)
    holding[40:80] = .95
    probability = np.column_stack([holding, np.zeros(len(time)), np.zeros(len(time))])
    if local_grasp:
        probability[40:48, 1] = .9
    probability[80:88, 2] = .9
    return {"trial": {"time": time, "events": np.array([.4, .8]),
                       "holding": (holding > .5).astype(float),
                       "pose_valid": np.zeros(len(time), bool),
                       "position": np.zeros((len(time), 3))},
            "prob": probability, "valid": np.ones(len(time), bool),
            "position": np.zeros((len(time), 3))}


def test_decoder_calibration_and_application_are_causal():
    items = [item()]
    holding = {"low": .35, "high": .65, "persistence_s": .03}

    def score(candidate, event):
        detected = candidate[0]["detections"][event]
        return float(bool(detected) and abs(detected[0] - [.4, .8][event]) <= .2)

    decoder = calibrate_decoder(items, holding, score)
    decoded = apply_decoder(items, decoder)[0]
    assert decoder["validation_event_macro_f1"] == 1.
    assert decoded["detections"][0]
    assert decoded["detections"][1]
    # Holding transition requires persistence and is never backdated.
    transition_only = dict(decoder)
    transition_only["events"] = [dict(x, local_weight=0.) for x in decoder["events"]]
    transitioned = apply_decoder(items, transition_only)[0]
    assert transitioned["detections"][0][0] >= .43
    assert transitioned["detections"][1][0] >= .83


def test_hybrid_training_and_validation_calibration_smoke(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "data"
    root.mkdir()
    for index in range(20):
        data = frame()
        data["EMG 1_S0"] += index * .01
        data.to_csv(root / f"trial_{index:03}.csv", index=False)
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["train", "--root", str(root),
        "--output-dir", str(output), "--device", "cpu", "--epochs", "1",
        "--raw-rate-hz", "1000", "--models", "emg", "--batch-size", "4"])
    original_main = trainer.train.main

    def small_main():
        trainer.train.MODEL_CLASS = ReachGraspHybrid
        trainer.train.MODEL_FORMAT = "reach_grasp_hybrid_v1"
        trainer.train.MODEL_EXTRA_ARGS = {"width": 16, "layers": 1, "heads": 2,
                                          "patch": 8, "stride": 2}
        original_main()

    monkeypatch.setattr(trainer.train, "main", small_main)
    trainer.main()
    state = torch.load(output / "emg_best.pt", weights_only=False)
    results = json.loads((output / "results.json").read_text())
    assert state["format"] == "reach_grasp_hybrid_v1"
    assert state["hybrid_event_decoder"]["events"]
    assert "combined" in results["hybrid_event_decoding"]["emg"]
