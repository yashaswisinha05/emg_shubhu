from __future__ import annotations

import inspect
from pathlib import Path

import torch

from scripts import train_latent_distillation_model as base
from scripts import train_channel_horizon_distillation_model as enhanced
from tests.test_latent_distillation import _config, _window
from emg_touch.models.channel_horizon_distillation import (
    ChannelHorizonLatentDistillationModel,
    EMGChannelTimeGate,
    channel_attention_regularizers,
    horizon_guidance_losses,
    physical_sensor_feature_indices,
)


def _enhanced_config() -> dict:
    config = _config(factor_guidance=True)
    config["data"] = {
        "sample_rate_hz": 100.0,
        "decimation": 1,
        "sensors": ["S0", "S4", "S8", "S12"],
    }
    config["model"]["latent_dim"] = 8
    config["model"]["factor_latent"].update({
        "intent_dim": 3,
        "motion_dim": 3,
    })
    config["model"]["channel_time_attention"] = {
        "hidden": 8,
        "temperature": 1.0,
        "sensor_dropout_probability": 0.25,
        "entropy_weight": 0.01,
        "smoothness_weight": 0.05,
        "report_lag_edges_ms": [0, 100, 250],
    }
    config["model"]["horizon_latent"] = {
        "start": 4,
        "dim": 2,
        "bins_ms": [50, 100, 200, 400],
        "head_width": 8,
        "gradient_reversal_scale": 0.25,
        "target_sigma_ms": 40.0,
        "regression_huber_beta": 0.1,
        "teacher_classification_weight": 0.25,
        "teacher_regression_weight": 0.25,
        "teacher_adversarial_weight": 0.025,
        "student_classification_weight": 0.5,
        "student_regression_weight": 0.5,
        "student_adversarial_weight": 0.05,
        "emg_classification_weight": 0.75,
        "emg_regression_weight": 0.75,
        "emg_adversarial_weight": 0.05,
    }
    return config


def test_channel_gate_starts_as_identity_and_reports_probabilities() -> None:
    gate = EMGChannelTimeGate(12, 4, hidden=8).eval()
    emg = torch.randn(2, 9, 12)
    mask = torch.ones(2, 9, dtype=torch.bool)
    gated, attention = gate(emg, mask)
    torch.testing.assert_close(gated, emg)
    torch.testing.assert_close(
        attention.sum(dim=-1), torch.ones(2, 9)
    )
    torch.testing.assert_close(
        attention, torch.full_like(attention, 0.25)
    )


def test_physical_sensor_ablation_selects_every_feature_view() -> None:
    assert physical_sensor_feature_indices(48, 4, 0) == list(range(0, 48, 4))
    assert physical_sensor_feature_indices(48, 4, 3) == list(range(3, 48, 4))
    all_indices = {
        index
        for sensor in range(4)
        for index in physical_sensor_feature_indices(48, 4, sensor)
    }
    assert all_indices == set(range(48))


def test_horizon_and_channel_losses_reach_the_emg_encoder() -> None:
    torch.manual_seed(11)
    config = _enhanced_config()
    model = ChannelHorizonLatentDistillationModel(config, 4, 6).train()
    window = _window()
    with torch.no_grad():
        teacher = model.teacher_forward(window["teacher_features"], sample=False)
    outputs = model.student_forward(
        window["emg"],
        window["imu"],
        window["time_mask"],
        sample=False,
        include_emg_only=True,
        apply_imu_dropout=True,
    )
    result = enhanced.student_objective(outputs, teacher, window, config)
    for name in (
        "horizon_classification",
        "horizon_regression",
        "horizon_adversarial",
        "emg_horizon_classification",
        "channel_entropy",
        "channel_smoothness",
    ):
        assert name in result
        assert torch.isfinite(result[name])
    result["loss"].backward()
    gradients = [
        parameter.grad
        for parameter in model.student.emg_encoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert sum(float(gradient.norm()) for gradient in gradients) > 0.0
    horizon_gradient = model.guidance.horizon.from_horizon[-1].weight.grad
    assert horizon_gradient is not None
    assert float(horizon_gradient.norm()) > 0.0


def test_horizon_supervision_is_a_label_not_a_student_input() -> None:
    config = _enhanced_config()
    model = ChannelHorizonLatentDistillationModel(config, 4, 6)
    parameters = set(inspect.signature(model.student_forward).parameters)
    assert not {
        "lead_samples", "position", "velocity", "trajectory_features"
    } & parameters

    window = _window()
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"],
        sample=False,
    )
    losses = horizon_guidance_losses(
        outputs, window["lead_samples"], 100.0,
        config["model"]["horizon_latent"],
    )
    assert all(torch.isfinite(value) for value in losses.values())


def test_channel_regularizers_ignore_left_padding() -> None:
    attention = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    ]])
    mask = torch.tensor([[False, False, True, True]])
    losses = channel_attention_regularizers(attention, mask)
    torch.testing.assert_close(losses["channel_entropy"], torch.tensor(1.0))
    torch.testing.assert_close(losses["channel_smoothness"], torch.tensor(0.0))


def test_new_entrypoint_does_not_import_virtual_leader_objective() -> None:
    source = Path(enhanced.__file__).read_text(encoding="utf-8")
    assert "virtual_leader_regularization import" not in source
    assert "train_virtual_leader_distillation_model" not in source


def test_original_objective_remains_callable_for_old_model() -> None:
    # The dedicated script keeps references rather than editing the baseline
    # implementation, so the previous 228.6 px experiment remains runnable.
    assert enhanced._BASE_STUDENT_OBJECTIVE is base.student_objective
