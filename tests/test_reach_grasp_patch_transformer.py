import json
import sys

import torch

from emg_touch.models.reach_grasp_patch_transformer import ReachGraspPatchTransformer
from scripts import train_reach_grasp_patch_transformer as trainer
from tests.test_reach_grasp import frame


def test_shape_causality_and_modality_isolation():
    torch.set_num_threads(1)
    model = ReachGraspPatchTransformer(width=16, layers=1, heads=2, patch=8, stride=2).eval()
    emg, imu = torch.randn(2, 50, 16), torch.randn(2, 50, 48)
    first = model(emg, imu)
    assert first["logits"].shape == (2, 50, 3)
    assert first["position"].shape == (2, 50, 3)
    torch.testing.assert_close(first["fusion_weights"].sum(-1), torch.ones(2, 50))
    emg[:, 30:] += 1000
    imu[:, 30:] -= 1000
    after = model(emg, imu)
    torch.testing.assert_close(first["logits"][:, :30], after["logits"][:, :30], atol=1e-5, rtol=1e-5)
    emg_only = ReachGraspPatchTransformer("emg", width=16, layers=1, heads=2,
                                          patch=8, stride=2).eval()
    before = emg_only(emg, imu)["logits"]
    after = emg_only(emg, imu + 10000)["logits"]
    torch.testing.assert_close(before, after)


def test_patch_transformer_training_smoke(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "data"
    root.mkdir()
    for i in range(20):
        f = frame()
        f["EMG 1_S0"] += i * .01
        f.to_csv(root / f"trial_{i:03}.csv", index=False)
    out = tmp_path / "out"
    # Keep this integration test small while exercising the wrapper/factory.
    monkeypatch.setattr(trainer.train, "MODEL_EXTRA_ARGS", {})
    monkeypatch.setattr(sys, "argv", ["train", "--root", str(root), "--output-dir", str(out),
        "--device", "cpu", "--epochs", "1", "--raw-rate-hz", "1000", "--models", "emg"])
    original_main = trainer.train.main
    def small_main():
        trainer.train.MODEL_CLASS = ReachGraspPatchTransformer
        trainer.train.MODEL_FORMAT = "reach_grasp_patch_transformer_v1"
        trainer.train.MODEL_EXTRA_ARGS = {"width": 16, "layers": 1, "heads": 2,
                                          "patch": 8, "stride": 2}
        original_main()
    monkeypatch.setattr(trainer.train, "main", small_main)
    trainer.main()
    state = torch.load(out / "emg_best.pt", weights_only=False)
    assert state["format"] == "reach_grasp_patch_transformer_v1"
    assert state["model_args"]["patch"] == 8
    assert "emg" in json.loads((out / "results.json").read_text())
