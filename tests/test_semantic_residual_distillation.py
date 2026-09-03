from __future__ import annotations

import inspect

import torch

from scripts import train_semantic_residual_distillation_model as semantic
from tests.test_channel_horizon_distillation import _enhanced_config
from tests.test_latent_distillation import _window
from emg_touch.models.semantic_residual_distillation import (
    SemanticResidualDistillationModel,
)


def _semantic_config() -> dict:
    config = _enhanced_config()
    config["model"]["semantic_residual"] = {
        "head_width": 8,
        "maximum_logit_delta": 1.5,
        "contrastive_temperature": 0.2,
        "fused_cosine_weight": 0.25,
        "emg_cosine_weight": 0.5,
        "fused_contrastive_weight": 0.25,
        "emg_contrastive_weight": 0.5,
        "fused_relational_weight": 0.1,
        "emg_relational_weight": 0.2,
        "endpoint_residual_weight": 1.0,
        "emg_endpoint_residual_weight": 0.75,
        "teacher_target_mix": 0.25,
        "teacher_advantage_temperature_px": 40.0,
        "teacher_advantage_weight": 1.0,
    }
    return config


def test_zero_initialized_residual_preserves_parent_predictions() -> None:
    model = SemanticResidualDistillationModel(_semantic_config(), 4, 6).eval()
    window = _window()
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"],
        sample=False, include_emg_only=True,
    )
    torch.testing.assert_close(outputs["prediction"], outputs["base_prediction"])
    torch.testing.assert_close(
        outputs["emg_only"]["prediction"],
        outputs["emg_only"]["base_prediction"],
    )
    assert outputs["residual_logit_delta"].abs().max() == 0


def test_semantic_losses_train_residual_and_emg_encoder() -> None:
    torch.manual_seed(7)
    config = _semantic_config()
    model = SemanticResidualDistillationModel(config, 4, 6).train()
    window = _window()
    with torch.no_grad():
        teacher = model.teacher_forward(window["teacher_features"], sample=False)
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"],
        sample=False, include_emg_only=True,
    )
    losses = semantic.student_objective(outputs, teacher, window, config)
    for name in (
        "semantic_fused_cosine",
        "semantic_emg_contrastive",
        "semantic_fused_relational",
        "endpoint_residual",
        "emg_endpoint_residual",
        "teacher_usefulness",
    ):
        assert name in losses
        assert torch.isfinite(losses[name])
    losses["loss"].backward()
    residual_gradient = (
        model.student.fused_endpoint_residual.network[-1].weight.grad
    )
    assert residual_gradient is not None
    assert float(residual_gradient.norm()) > 0.0
    emg_gradients = [
        parameter.grad for parameter in model.student.emg_encoder.parameters()
        if parameter.grad is not None
    ]
    assert sum(float(gradient.norm()) for gradient in emg_gradients) > 0.0


def test_student_deployment_api_has_no_teacher_or_vive_input() -> None:
    parameters = set(
        inspect.signature(
            SemanticResidualDistillationModel.student_forward
        ).parameters
    )
    forbidden = {
        "position", "velocity", "teacher_features", "trajectory_features",
        "lead_samples", "target",
    }
    assert not (parameters & forbidden)


def test_target_aware_contrastive_treats_same_target_as_positive() -> None:
    intent = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    aligned = semantic.target_aware_contrastive_loss(
        intent, intent, labels, temperature=0.1
    )
    shuffled = semantic.target_aware_contrastive_loss(
        intent[[2, 0, 1]], intent, labels, temperature=0.1
    )
    assert aligned < shuffled
