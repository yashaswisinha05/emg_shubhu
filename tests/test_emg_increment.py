import copy
import numpy as np
import torch

from scripts.diagnose_emg_increment import inspect_group, summarize, load_model
from scripts.train_emg_delay_correction import Correction
from emg_touch.models.imu_to_emg import WearableTaskNetwork


def test_bootstrap_clusters_repeated_leads():
    rows = []
    for trial, gain in [("a", 1.), ("b", 3.)]:
        for lead in [0, 50]:
            rows.append({"trial": trial, "lead_ms": lead, "emg": [10., 2., 3.],
                         **{name: [10. + gain, 3., 4.] for name in
                            ["imu_baseline", "imu_control", "zero_emg", "shuffled_emg"]}})
    report = summarize(rows, 100, 1)
    assert report["trials"] == 2 and report["observations"] == 4
    assert report["gains"]["imu_control_minus_emg"]["screen_px"]["mean"] == 2.
    doubled = summarize(rows + rows, 100, 1)
    assert doubled["gains"] == report["gains"]


def test_group_inference_and_singleton_final_batch(tmp_path):
    torch.set_num_threads(1)
    config = {"data": {"sample_rate_hz": 1000, "decimation": 1},
              "imu_to_emg": {"context_ms": 10, "steps": 4, "cutoffs": 1,
                             "lead_ms": [0, 5]}}
    args = dict(input_dim=2, hidden=8, latent=4, steps=4)
    teacher = WearableTaskNetwork(**args)
    emg = Correction(teacher, 1).eval()
    control = Correction(copy.deepcopy(teacher), 1, control=True).eval()
    path = tmp_path / "selected.pt"
    torch.save({"format": "emg_delay_correction_v1", "teacher_args": args,
                "emg_dim": 1, "control": False, "state_dict": emg.state_dict()}, path)
    emg, _ = load_model(path, "cpu")
    batch = {"emg": torch.randn(3, 30, 1), "imu": torch.randn(3, 30, 2),
             "position": torch.zeros(3, 30, 3), "velocity": torch.zeros(3, 30, 3),
             "onset": torch.zeros(3, dtype=torch.long), "lengths": torch.tensor([30, 30, 30]),
             "screen_target": torch.rand(3, 2),
             "canvas": torch.tensor([[1920., 1080.]]).repeat(3, 1),
             "paths": ["recording/a.csv", "recording/b.csv", "recording/c.csv"]}
    rows = inspect_group(emg, control, [batch], config, "cpu", 5, 0, "recording", 2)
    assert len(rows) == 3
    assert all(r["trial"] != r["shuffle_donor"] for r in rows)
    assert all(r["shuffle_donor"].startswith("recording/") for r in rows)
    assert np.isfinite(rows[0]["emg"]).all()
    # Zero-initialized correction exactly matches the original teacher.
    for row in rows:
        np.testing.assert_allclose(row["emg"], row["imu_baseline"])
