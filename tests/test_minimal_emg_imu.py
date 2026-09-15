import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from minimal_emg_imu.model import MinimalEMGIMUModel
from minimal_emg_imu.train import minimal_loss


def small_model():
    return MinimalEMGIMUModel(
        width=16, patch=8, stride=4, layers=1, heads=4,
        dropout=0., react_context=20, adapter_layers=1, future_steps=20).eval()


def test_minimal_model_outputs_only_selected_tasks_and_diagnostics():
    model = small_model()
    output = model(torch.randn(2, 32, 16), torch.randn(2, 32, 48))
    assert output["gripper_state_logits"].shape == (2, 32, 2)
    assert output["position"].shape == (2, 32, 3)
    assert output["click"].shape == (2, 32, 2)
    assert output["future_position"].shape == (2, 32, 20, 3)
    assert output["orientation_6d"] is None
    assert output["final_position"] is None


def test_minimal_model_is_causal():
    torch.manual_seed(4)
    model = small_model()
    emg, imu = torch.randn(1, 36, 16), torch.randn(1, 36, 48)
    changed_emg, changed_imu = emg.clone(), imu.clone()
    changed_emg[:, 24:] = torch.randn_like(changed_emg[:, 24:])
    changed_imu[:, 24:] = torch.randn_like(changed_imu[:, 24:])
    first = model(emg, imu)
    second = model(changed_emg, changed_imu)
    for key in ("gripper_state_logits", "position", "click", "future_position"):
        assert torch.allclose(first[key][:, :24], second[key][:, :24], atol=1e-6)


def test_minimal_loss_uses_only_four_selected_targets():
    model = small_model()
    emg, imu = torch.randn(2, 32, 16), torch.randn(2, 32, 48)
    output = model(emg, imu)
    batch = {
        "emg_usable": torch.ones(2, 32, dtype=torch.bool),
        "imu_usable": torch.ones(2, 32, dtype=torch.bool),
        "gripper_state_valid": torch.ones(2, 32, dtype=torch.bool),
        "gripper_state": torch.randint(0, 2, (2, 32)),
        "pose_mask": torch.ones(2, 32, 1),
        "pose": torch.randn(2, 32, 3),
        "click_valid": torch.ones(2, 32, dtype=torch.bool),
        "click_target": torch.rand(2, 32, 2),
        "canvas_px": torch.tensor([[1920., 1080.], [1920., 1080.]]),
        "trial_progress": torch.linspace(0., 1., 32).expand(2, -1),
    }
    args = SimpleNamespace(
        state_weight=1., position_weight=1., pixel_weight=.35,
        future_pose_weight=.5)
    value = minimal_loss(model, output, batch, torch.ones(2), args)
    assert value.ndim == 0 and torch.isfinite(value)
    value.backward()
    for head in (model.gripper_context, model.position_head[-1],
                 model.pixel_head[-1], model.future_head[-1]):
        assert head.weight.grad is not None


def test_minimal_entry_points_have_focused_help():
    root = Path(__file__).resolve().parents[1]
    for script in ("minimal_emg_imu/train.py", "minimal_emg_imu/infer.py"):
        result = subprocess.run(
            [sys.executable, script, "--help"], cwd=root,
            capture_output=True, text=True)
        assert result.returncode == 0
