import copy
from types import SimpleNamespace

import numpy as np
import torch

from emg_touch.models.imu_to_emg import WearableTaskNetwork, task_loss, transfer_loss
from scripts.train_complete_reach_model import make_complete_reach_window
from scripts.train_imu_to_emg import evaluate, fit


def batch():
    return {"emg": torch.randn(2, 24, 4), "imu": torch.randn(2, 24, 6),
            "position": torch.randn(2, 24, 3) * .03,
            "velocity": torch.zeros(2, 24, 3), "lengths": torch.tensor([24, 22]),
            "onset": torch.tensor([2, 3]), "screen_target": torch.rand(2, 2),
            "canvas": torch.tensor([[1920., 1080.], [1440., 900.]])}


def window(b):
    return make_complete_reach_window(b, 16, 1, 4, np.random.default_rng(1),
                                      .8, 1., fixed_lead=5)


def test_future_and_imu_do_not_enter_student():
    b = batch()
    w = window(b)
    model = WearableTaskNetwork(4, hidden=8, latent=4, steps=4).eval()
    reference = model(w["emg"], w["time_mask"])
    changed = copy.deepcopy(b)
    changed["imu"].fill_(1000)
    changed["position"].fill_(-1000)
    for row, length in enumerate(b["lengths"]):
        changed["emg"][row, int(length) - 5:] = 1000
    after = window(changed)
    actual = model(after["emg"], after["time_mask"])
    for key in reference:
        torch.testing.assert_close(actual[key], reference[key])


def test_padding_and_teacher_stop_gradient():
    student = WearableTaskNetwork(4, hidden=8, latent=4, steps=4).eval()
    teacher = WearableTaskNetwork(6, hidden=8, latent=4, steps=4).eval()
    w = window(batch())
    mask = w["time_mask"].clone()
    mask[:, :4] = False
    signal = w["emg"].clone()
    before = student(signal, mask)
    signal[:, :4] = 1e5
    after = student(signal, mask)
    torch.testing.assert_close(before["path"], after["path"])
    t = teacher(w["imu"], mask)
    loss = task_loss(after, w) + transfer_loss(after, t, w)
    loss.backward()
    assert any(p.grad is not None for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    assert torch.equal(after["path"][:, 0], torch.zeros(2, 3))


def test_training_checkpoint_and_evaluation_smoke(tmp_path):
    torch.set_num_threads(1)
    kwargs = {"hidden": 8, "latent": 4, "steps": 4}
    config = {"data": {"sample_rate_hz": 100., "decimation": 1},
              "imu_to_emg": {"context_ms": 160, "steps": 4, "cutoffs": 2,
                  "lead_ms": [0, 50], "evaluation_leads_ms": [0, 50],
                  "joint_px_per_cm": 5,
                  "model_args": {"imu": dict(input_dim=6, **kwargs),
                                 "emg": dict(input_dim=4, **kwargs)}}}
    args = SimpleNamespace(seed=3, lr=.001, output_dir=tmp_path, device="cpu",
                           patience=2, latent_weight=.1, output_weight=.25)
    loaders = ([batch()], [batch()], [batch()])
    teacher = WearableTaskNetwork(6, **kwargs)
    teacher = fit("imu_teacher", teacher, "imu", None, loaders, config, args, 1)
    teacher.requires_grad_(False)
    original = copy.deepcopy(teacher.state_dict())
    student = WearableTaskNetwork(4, **kwargs)
    fit("emg_baseline", copy.deepcopy(student), "emg", None, loaders, config, args, 1)
    student = fit("emg_student", student, "emg", teacher, loaders, config, args, 1)
    for key, value in original.items():
        torch.testing.assert_close(teacher.state_dict()[key], value)
    saved = torch.load(tmp_path / "emg_student.pt", weights_only=False)
    restored = WearableTaskNetwork(**saved["model_args"])
    restored.load_state_dict(saved["state_dict"])
    assert saved["input_modality"] == "emg"
    result = evaluate(restored, "emg", loaders[2], config, "cpu")
    assert np.isfinite(result["overall"]["screen_px"])
    assert result["0"]["observations"] == 2
