import torch
from scripts.train_emg_delay_correction import Correction, delayed_windows
from emg_touch.models.imu_to_emg import WearableTaskNetwork, task_loss


def test_delay_and_frozen_zero_initialization():
    torch.set_num_threads(1)
    config = {"data": {"sample_rate_hz": 1000, "decimation": 1},
              "imu_to_emg": {"context_ms": 10, "steps": 4, "cutoffs": 1,
                             "lead_ms": [0, 0]}}
    b = {"emg": torch.arange(30.).view(1, 30, 1), "imu": torch.randn(1, 30, 2),
         "position": torch.zeros(1, 30, 3), "velocity": torch.zeros(1, 30, 3),
         "onset": torch.tensor([0]), "lengths": torch.tensor([30]),
         "screen_target": torch.zeros(1, 2), "canvas": torch.tensor([[1920., 1080.]])}
    w = next(delayed_windows([b], config, "cpu", 1, 5, 0))
    torch.testing.assert_close(w["emg"].flatten(), torch.arange(15., 25.))
    torch.testing.assert_close(w["imu"], b["imu"][:, -10:])
    teacher = WearableTaskNetwork(2, hidden=8, latent=4, steps=4).eval()
    model = Correction(teacher, 1)
    out = model(w)
    original = teacher(w["imu"], w["time_mask"])
    torch.testing.assert_close(out["screen"], original["screen"])
    torch.testing.assert_close(out["path"], original["path"])
    model.train()
    assert not teacher.training
    task_loss(out, w).backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert model.head[-1].weight.grad is not None


def test_control_ignores_emg():
    teacher = WearableTaskNetwork(2, hidden=8, latent=4, steps=4)
    model = Correction(teacher, 1, control=True).eval()
    torch.nn.init.normal_(model.head[-1].weight)
    w = {"imu": torch.randn(2, 8, 2), "time_mask": torch.ones(2, 8, dtype=torch.bool),
         "emg": torch.randn(2, 8, 1), "emg_mask": torch.ones(2, 8, dtype=torch.bool)}
    before = model(w)
    w["emg"].fill_(1000)
    after = model(w)
    torch.testing.assert_close(before["screen"], after["screen"])
