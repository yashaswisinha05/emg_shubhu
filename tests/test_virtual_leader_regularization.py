from __future__ import annotations

import torch

from scripts import train_latent_distillation_model as base
from scripts import train_virtual_leader_distillation_model as enhanced
from tests.test_latent_distillation import _config, _window
from emg_touch.models.latent_distillation import WearableLatentDistillationModel
from emg_touch.virtual_leader_regularization import (
    trajectory_kinematics,
    virtual_leader_losses,
)


def _settings() -> dict:
    return {
        "enabled": True,
        "velocity_scale_mps": 1.0,
        "acceleration_scale_mps2": 10.0,
        "huber_beta": 0.1,
        "normalized_residual_clip": 5.0,
        "mean_reversion_per_s2": 25.0,
        "drag_per_s": 10.0,
        "teacher_velocity_weight": 0.1,
        "teacher_acceleration_weight": 0.05,
        "teacher_endpoint_weight": 0.5,
        "teacher_dynamics_weight": 0.025,
        "student_velocity_weight": 0.25,
        "student_acceleration_weight": 0.1,
        "student_endpoint_weight": 1.0,
        "student_dynamics_weight": 0.05,
        "imu_velocity_weight": 0.1,
        "imu_acceleration_weight": 0.05,
        "imu_endpoint_weight": 0.25,
        "imu_dynamics_weight": 0.025,
    }


def test_kinematics_respects_each_trials_lead_time() -> None:
    trajectory = torch.tensor([
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
    ])
    velocity, acceleration, dt = trajectory_kinematics(
        trajectory, torch.tensor([10, 20]), sample_rate_hz=100.0
    )
    torch.testing.assert_close(dt.flatten(), torch.tensor([0.05, 0.10]))
    torch.testing.assert_close(velocity[0, :, 0], torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(velocity[1, :, 0], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(acceleration, torch.zeros_like(acceleration))


def test_matching_trajectory_has_zero_tracking_losses() -> None:
    target = torch.randn(2, 5, 3) * 0.05
    losses = virtual_leader_losses(
        target, target, torch.tensor([20, 30]), 100.0, _settings()
    )
    torch.testing.assert_close(losses["velocity"], torch.tensor(0.0))
    torch.testing.assert_close(losses["acceleration"], torch.tensor(0.0))
    torch.testing.assert_close(losses["endpoint"], torch.tensor(0.0))
    assert torch.isfinite(losses["dynamics"])


def test_virtual_leader_losses_backpropagate_to_predicted_trajectory() -> None:
    prediction = (torch.randn(2, 5, 3) * 0.05).requires_grad_()
    target = torch.randn(2, 5, 3) * 0.05
    losses = virtual_leader_losses(
        prediction, target, torch.tensor([20, 30]), 100.0, _settings()
    )
    sum(losses.values()).backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert float(prediction.grad.norm()) > 0.0


def test_enhanced_objective_adds_all_losses_without_tracker_inputs() -> None:
    torch.manual_seed(9)
    config = _config(factor_guidance=True)
    config["data"] = {"sample_rate_hz": 100.0, "decimation": 1}
    config["virtual_leader_regularization"] = _settings()
    model = WearableLatentDistillationModel(config, 4, 6).train()
    window = _window()
    with torch.no_grad():
        teacher = model.teacher_forward(window["teacher_features"], sample=False)
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"],
        sample=False, include_emg_only=True,
    )
    original = base.student_objective(outputs, teacher, window, config)
    result = enhanced.student_objective(outputs, teacher, window, config)
    assert float(result["loss"].detach()) > float(original["loss"].detach())
    for name in (
        "vl_velocity", "vl_acceleration", "vl_endpoint", "vl_dynamics",
        "imu_vl_velocity", "imu_vl_acceleration", "imu_vl_endpoint",
        "imu_vl_dynamics",
    ):
        assert name in result
        assert torch.isfinite(result[name])
    result["loss"].backward()
    gradients = [
        parameter.grad for parameter in model.student.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
