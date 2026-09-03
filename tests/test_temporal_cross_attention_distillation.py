from __future__ import annotations

import inspect

import numpy as np
import torch

from scripts import train_temporal_cross_attention_model as trainer
from tests.test_latent_distillation import _window
from tests.test_semantic_residual_distillation import _semantic_config
from emg_touch.live_distillation import checkpoint_kind
from emg_touch.models.temporal_cross_attention_distillation import (
    TemporalCrossAttentionDistillationModel,
)


def _config() -> dict:
    config = _semantic_config()
    config["model"]["temporal_cross_attention"] = {
        "num_heads": 2,
        "dropout": 0.0,
        "lag_hidden": 8,
        "lag_edges_ms": [0, 50, 100, 250],
        "lag_entropy_weight": 0.001,
        "fused_intent_alignment_weight": 0.5,
        "emg_intent_alignment_weight": 1.0,
    }
    config["distillation"]["latent_distillation_weight"] = 0.0
    config["distillation"]["emg_latent_weight"] = 0.0
    config["distillation"]["student_noise_scale"] = 0.0
    return config


def test_student_is_deterministic_and_reports_sensor_lag_attention() -> None:
    torch.manual_seed(3)
    model = TemporalCrossAttentionDistillationModel(_config(), 4, 6).eval()
    window = _window()
    first = model.student_forward(
        window["emg"], window["imu"], window["time_mask"],
        sample=True, noise_scale=100.0, include_emg_only=True,
    )
    second = model.student_forward(
        window["emg"], window["imu"], window["time_mask"],
        sample=True, noise_scale=100.0, include_emg_only=True,
    )
    torch.testing.assert_close(first["prediction"], second["prediction"])
    attention = first["lag_attention"]
    assert attention.shape == (2, 4, 3)
    torch.testing.assert_close(attention.sum(dim=(1, 2)), torch.ones(2))
    assert first["emg_from_imu_attention"].ndim == 4
    assert first["imu_from_emg_attention"].ndim == 4
    assert checkpoint_kind(model.state_dict()) == "temporal_cross_attention"


def test_intent_block_is_owned_by_emg_and_motion_block_by_imu() -> None:
    torch.manual_seed(4)
    config = _config()
    model = TemporalCrossAttentionDistillationModel(config, 4, 6).eval()
    window = _window()
    emg = window["emg"]
    imu = window["imu"]
    mask = window["time_mask"]
    base = model.student_forward(emg, imu, mask)["mu"]
    changed_imu = model.student_forward(emg, imu + 1000.0, mask)["mu"]
    changed_emg = model.student_forward(emg + 1000.0, imu, mask)["mu"]
    intent = int(config["model"]["factor_latent"]["intent_dim"])
    motion = int(config["model"]["factor_latent"]["motion_dim"])
    torch.testing.assert_close(base[:, :intent], changed_imu[:, :intent])
    torch.testing.assert_close(
        base[:, intent : intent + motion],
        changed_emg[:, intent : intent + motion],
    )
    assert not torch.equal(base[:, :intent], changed_emg[:, :intent])
    assert not torch.equal(
        base[:, intent : intent + motion],
        changed_imu[:, intent : intent + motion],
    )


def test_new_objective_reaches_cross_attention_and_lag_encoder() -> None:
    torch.manual_seed(5)
    config = _config()
    model = TemporalCrossAttentionDistillationModel(config, 4, 6).train()
    window = _window()
    with torch.no_grad():
        teacher = model.teacher_forward(window["teacher_features"], sample=False)
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"],
        include_emg_only=True, apply_imu_dropout=True,
    )
    losses = trainer.student_objective(outputs, teacher, window, config)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    for module in (
        model.student.emg_from_imu,
        model.student.imu_from_emg,
        model.student.lag_attention,
    ):
        gradients = [
            parameter.grad for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert sum(float(gradient.norm()) for gradient in gradients) > 0.0


def test_deployment_api_contains_no_tracker_or_true_horizon() -> None:
    parameters = set(
        inspect.signature(
            TemporalCrossAttentionDistillationModel.student_forward
        ).parameters
    )
    assert not {
        "position", "velocity", "teacher_features", "trajectory_features",
        "target", "lead_samples",
    } & parameters


def test_duration_bins_answer_slow_movement_without_test_selection() -> None:
    durations = np.asarray([300, 100, 600, 200, 500, 400], dtype=np.float64)
    errors = np.asarray([30, 10, 60, 20, 50, 40], dtype=np.float64)
    result = trainer.duration_bin_summary(durations, errors)
    assert result["duration_fast_mean_ms"] == 150.0
    assert result["duration_slow_mean_ms"] == 550.0
    assert result["duration_slow_minus_fast_px"] == 40.0
