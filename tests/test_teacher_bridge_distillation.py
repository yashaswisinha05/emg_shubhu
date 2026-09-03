from __future__ import annotations

import inspect

import torch

from scripts import train_teacher_bridge_model as trainer
from tests.test_latent_distillation import _window
from tests.test_temporal_cross_attention_distillation import _config as _temporal_config
from emg_touch.live_distillation import checkpoint_kind
from emg_touch.models.teacher_bridge_distillation import (
    TeacherBridgeDistillationModel,
)


def _config() -> dict:
    config = _temporal_config()
    config["model"]["teacher_bridge"] = {
        "hidden": 8,
        "dropout": 0.0,
        "temperature": 2.0,
        "advantage_temperature_px": 40.0,
        "minimum_teacher_weight": 0.25,
        "output_distillation_weight": 1.0,
        "emg_output_distillation_weight": 0.5,
        "heatmap_weight": 1.0,
        "offset_weight": 0.5,
        "direct_weight": 1.0,
        "decoder_latent_weight": 0.1,
    }
    return config


def test_student_decoder_is_independent_and_refreshes_from_teacher() -> None:
    model = TeacherBridgeDistillationModel(_config(), 4, 6)
    teacher_parameter = next(model.decoder.parameters())
    student_parameter = next(model.student.endpoint_decoder.parameters())
    assert teacher_parameter is not student_parameter
    with torch.no_grad():
        teacher_parameter.add_(1.0)
    assert not torch.equal(teacher_parameter, student_parameter)
    model.initialise_student_decoder_from_teacher()
    torch.testing.assert_close(teacher_parameter, student_parameter)


def test_hierarchical_distillation_trains_bridge_and_student_decoder_only() -> None:
    torch.manual_seed(8)
    config = _config()
    model = TeacherBridgeDistillationModel(config, 4, 6).train()
    model.initialise_student_decoder_from_teacher()
    window = _window()
    with torch.no_grad():
        teacher = model.teacher_forward(window["teacher_features"], sample=False)
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"],
        include_emg_only=True, apply_imu_dropout=True,
    )
    losses = trainer.student_objective(outputs, teacher, window, config)
    for name in (
        "teacher_heatmap", "teacher_offset", "teacher_direct",
        "teacher_decoder_latent", "teacher_usefulness",
    ):
        assert name in losses
        assert torch.isfinite(losses[name])
    losses["loss"].backward()
    bridge_gradient = (
        model.student.teacher_latent_bridge.network[-1].weight.grad
    )
    decoder_gradient = next(
        parameter.grad for parameter in model.student.endpoint_decoder.parameters()
        if parameter.grad is not None
    )
    assert float(bridge_gradient.norm()) > 0.0
    assert float(decoder_gradient.norm()) > 0.0
    assert all(parameter.grad is None for parameter in model.decoder.parameters())


def test_bridge_checkpoint_and_deployment_boundary() -> None:
    model = TeacherBridgeDistillationModel(_config(), 4, 6).eval()
    assert checkpoint_kind(model.state_dict()) == "teacher_bridge"
    parameters = set(inspect.signature(model.student_forward).parameters)
    assert not {
        "position", "velocity", "teacher_features", "trajectory_features",
        "target", "lead_samples",
    } & parameters
    window = _window()
    first = model.student_forward(
        window["emg"], window["imu"], window["time_mask"],
        sample=True, noise_scale=100.0,
    )
    second = model.student_forward(
        window["emg"], window["imu"], window["time_mask"],
        sample=True, noise_scale=100.0,
    )
    torch.testing.assert_close(first["prediction"], second["prediction"])
    assert "base_prediction" not in first
