import numpy as np
import torch

from emg_touch.data.annotation_uncertainty import soften_events, holding_transitions
from scripts import train_reach_grasp as trainer


def test_soft_targets_and_boundary_mask():
    trial = {"time": np.arange(101) / 100, "events": np.array([.3, .7])}
    soften_events(trial, .2)
    assert trial["event_labels"][30, 0] == 1
    assert 0 < trial["event_labels"][20, 0] < 1
    assert trial["event_labels"][0, 0] == 0
    assert not trial["holding_certain"][30]
    assert trial["holding_certain"][0]


def test_causal_transition_latency_and_gap():
    times = np.arange(100) / 100
    prob = np.zeros(100)
    prob[30:70] = 1
    valid = np.ones(100, bool)
    found = holding_transitions(times, prob, valid, persistence_s=.05)
    np.testing.assert_allclose(found, [[.35], [.75]])
    assert holding_transitions(times[:60], prob[:60], valid[:60], persistence_s=.05)[0] == found[0]
    # Initial holding state is not invented as an observed onset.
    assert holding_transitions(times, np.ones(100), valid)[0] == []
    valid[32:40] = False
    found = holding_transitions(times, prob, valid, persistence_s=.05)
    np.testing.assert_allclose(found[0], [.45])


def test_soft_training_smoke(tmp_path, monkeypatch):
    from tests.test_reach_grasp import frame
    import sys
    import json
    torch.set_num_threads(1)
    root = tmp_path / "data"
    root.mkdir()
    for i in range(20):
        f = frame()
        f["EMG 1_S0"] += i * .01
        f.to_csv(root / f"trial_{i:03}.csv", index=False)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["train", "--root", str(root), "--output-dir", str(out),
        "--device", "cpu", "--epochs", "1", "--raw-rate-hz", "1000", "--annotation-aware"])
    trainer.main()
    results = json.loads((out / "results.json").read_text())
    assert set(results["emg+imu"]["by_tolerance_ms"]) == {
        "100", "150", "200", "300", "1500"}
    assert results["emg"]["holding_boundary_excluded"]
    state = torch.load(out / "emg_imu_best.pt", weights_only=False)
    assert state["format"] == "reach_grasp_annotation_v1"
    assert "holding_decoder" in state
