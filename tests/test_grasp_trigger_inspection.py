import json
import sys

import numpy as np
import torch

from scripts import inspect_grasp_triggers as inspect
from scripts import train_reach_grasp as train
from tests.test_reach_grasp import frame, settings
from emg_touch.data.reach_grasp import preprocess
from emg_touch.data.annotation_uncertainty import soften_events
from emg_touch.models.reach_grasp import ReachGraspModel


def test_confirmation_is_causal_and_rearms():
    t = np.arange(150) / 100
    p = np.zeros(150)
    p[20:40] = 1
    p[90:110] = 1
    valid = np.ones(150, bool)
    found = inspect.stable_triggers(t, p, valid, .5, persistence_s=.06)
    np.testing.assert_allclose(found, [.26, .96])
    assert inspect.stable_triggers(t[:70], p[:70], valid[:70], .5, persistence_s=.06) == found[:1]
    valid[23:30] = False
    np.testing.assert_allclose(inspect.stable_triggers(t, p, valid, .5, persistence_s=.06), [.36, .96])


def test_inspection_smoke(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    run = tmp_path / "run"
    run.mkdir()
    files = []
    for i in range(4):
        path = tmp_path / f"trial_{i:03}.csv"
        frame().to_csv(path, index=False)
        files.append(str(path))
    (run / "splits.json").write_text(json.dumps({"train": files[:1], "validation": files[1:3], "test": files[3:]}))
    prep = dict(settings(), annotation_uncertainty_s=.2)
    norm = train.normalization([preprocess(files[0], prep)])
    for mode in ["imu", "emg", "emg+imu"]:
        model = ReachGraspModel(mode)
        torch.save({"format": "reach_grasp_annotation_v1", "model_args": {"modality": mode},
                    "state_dict": model.state_dict(), "preprocessing": prep, "normalization": norm,
                    "event_thresholds": [.5, .5], "event_tolerance_s": .2,
                    "holding_decoder": {"low": .3, "high": .7, "persistence_s": .06}},
                   run / (mode.replace("+", "_") + "_best.pt"))
    out = tmp_path / "inspection"
    monkeypatch.setattr(sys, "argv", ["inspect", "--run-dir", str(run), "--output-dir", str(out),
                                     "--device", "cpu", "--plots", "1"])
    inspect.main()
    assert len(list(out.glob("*.png"))) == 3
    r = json.loads((out / "decoder_report.json").read_text())
    for name in ["imu", "emg", "emg_imu"]:
        baseline = r[name]["validation_sweep"][0]["metrics"]["event_macro_f1"]
        selected = r[name]["by_tolerance_ms"]["200"]["selected_heads"]["event_macro_f1"]
        assert selected >= baseline
    print("PREVIEW", next(out.glob("emg_imu*.png")))
