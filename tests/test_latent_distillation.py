from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from scripts.train_latent_distillation_model import (
    AdaptiveTrialDifficulty,
    make_distillation_window,
    select_sessions,
    student_objective,
)
from emg_touch.models.latent_distillation import (
    WearableLatentDistillationModel,
    diagonal_gaussian_kl,
)


def _config(*, imu_dropout: float = 0.0) -> dict:
    return {
        "model": {
            "grid_size": [2, 2],
            "d_model": 8,
            "num_layers": 1,
            "num_heads": 2,
            "ffn_dim": 16,
            "dropout": 0.0,
            "patch_length": 4,
            "patch_stride": 2,
            "tcn_kernel_sizes": [3],
            "latent_dim": 4,
            "teacher_layers": 1,
            "decoder_width": 8,
            "teacher_trajectory_steps": 4,
            "trajectory_limit_m": 1.0,
            "velocity_scale_mps": 1.0,
            "imu_modality_dropout": imu_dropout,
        },
        "loss": {
            "gaussian_sigma_cells": 0.5,
            "gaussian_soft_fraction": 0.15,
            "edge_weight": 0.0,
            "heatmap_weight": 1.0,
            "offset_weight": 0.5,
            "pixel_weight": 1.0,
            "radial_weight": 0.5,
            "transport_weight": 0.5,
            "offset_huber_beta": 0.1,
            "pixel_huber_beta": 0.25,
            "pixel_normalizer_px": 80.0,
            "charbonnier_epsilon_px": 1.0,
        },
        "distillation": {
            "trajectory_epsilon_m": 0.002,
            "teacher_sigma_floor": 0.05,
            "student_trajectory_weight": 2.0,
            "latent_distillation_weight": 1.0,
            "prediction_distillation_weight": 0.25,
            "trajectory_distillation_weight": 1.0,
            "emg_only_weight": 0.5,
            "emg_latent_weight": 0.5,
            "imu_tracking_weight": 0.25,
        },
    }


def _batch() -> dict[str, object]:
    steps = 40
    time = torch.arange(steps, dtype=torch.float32).view(1, steps, 1)
    position = 0.01 * time.expand(2, -1, 3).clone()
    return {
        "emg": time.expand(2, -1, 4).clone(),
        "imu": (10.0 * time).expand(2, -1, 6).clone(),
        "position": position,
        "velocity": torch.full_like(position, 0.01),
        "lengths": torch.tensor([steps, steps]),
        "onset": torch.tensor([10, 15]),
        "screen_target": torch.tensor([[0.2, 0.3], [0.7, 0.8]]),
        "canvas": torch.tensor([[1000.0, 500.0], [1000.0, 500.0]]),
        "paths": ["easy.csv", "hard.csv"],
    }


def _window(batch: dict[str, object] | None = None) -> dict[str, object]:
    result = make_distillation_window(
        _batch() if batch is None else batch,
        context_samples=24,
        patch_length=4,
        teacher_steps=4,
        generator=np.random.default_rng(0),
        trajectory_limit_m=1.0,
        velocity_scale_mps=1.0,
        fixed_lead=8,
    )
    assert result is not None
    return result


def test_student_window_is_causal_and_teacher_target_is_future_vive() -> None:
    window = _window()
    assert window["emg"].shape == (2, 24, 4)
    # touch=39 and lead=8 => cut=31, so sample 30 is the final input.
    assert window["emg"][:, -1, 0].tolist() == [30.0, 30.0]
    assert window["time_mask"].sum(dim=1).tolist() == [24, 24]
    # The final privileged target is touch position relative to sample 30.
    torch.testing.assert_close(
        window["trajectory_target"][:, -1],
        torch.full((2, 3), 0.09),
    )


def test_tracker_changes_cannot_change_student_inputs_or_api() -> None:
    original = _batch()
    changed = _batch()
    changed["position"] = changed["position"].clone()
    changed["velocity"] = changed["velocity"].clone()
    changed["position"][:, 31:] += 1000.0
    changed["velocity"][:, 31:] -= 1000.0
    first, second = _window(original), _window(changed)
    torch.testing.assert_close(first["emg"], second["emg"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first["imu"], second["imu"], rtol=0.0, atol=0.0)
    assert not torch.equal(first["teacher_features"], second["teacher_features"])

    parameters = set(
        inspect.signature(WearableLatentDistillationModel.student_forward).parameters
    )
    assert not ({"position", "velocity", "trajectory_features"} & parameters)


def test_student_objective_reaches_both_wearable_encoders() -> None:
    torch.manual_seed(1)
    config = _config()
    model = WearableLatentDistillationModel(config, 4, 6).train()
    window = _window()
    with torch.no_grad():
        teacher = model.teacher_forward(window["teacher_features"], sample=False)
    student = model.student_forward(
        window["emg"], window["imu"], window["time_mask"],
        sample=False, include_emg_only=True,
    )
    loss = student_objective(student, teacher, window, config)["loss"]
    loss.backward()
    for encoder in (model.student.emg_encoder, model.student.imu_encoder):
        gradients = [
            parameter.grad for parameter in encoder.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.norm()) for gradient in gradients) > 0.0


def test_full_imu_dropout_removes_imu_from_fused_latent() -> None:
    torch.manual_seed(2)
    model = WearableLatentDistillationModel(_config(imu_dropout=1.0), 4, 6).train()
    emg = torch.randn(2, 24, 4)
    mask = torch.ones(2, 24, dtype=torch.bool)
    first = model.student_forward(
        emg, torch.randn(2, 24, 6), mask,
        sample=False, apply_imu_dropout=True,
    )
    second = model.student_forward(
        emg, 100.0 * torch.randn(2, 24, 6), mask,
        sample=False, apply_imu_dropout=True,
    )
    torch.testing.assert_close(first["mu"], second["mu"], rtol=0.0, atol=0.0)


def test_gaussian_distillation_is_zero_for_identical_distributions() -> None:
    mu = torch.randn(3, 4)
    log_variance = torch.randn(3, 4).clamp(-4.0, 1.0)
    loss = diagonal_gaussian_kl(mu, log_variance, mu, log_variance)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_adaptive_sampler_upweights_hard_trials_with_a_cap() -> None:
    memory = AdaptiveTrialDifficulty(
        alpha=1.0, uniform_mix=1.0, power=1.0, max_ratio=4.0
    )
    memory.update(["easy.csv", "hard.csv"], torch.tensor([10.0, 1000.0]))
    weights = memory.weights_for(["easy.csv", "hard.csv"])
    assert weights[1] > weights[0]
    assert float(weights.max()) <= 4.0


def test_session_selection_keeps_only_requested_dev_a_folders() -> None:
    sessions = {
        "dev_a1_vive__first": [Path("a1/trial_1.csv")],
        "dev_a2_vive__second": [Path("a2/trial_1.csv")],
        "dev_a3_vive__third": [Path("a3/trial_1.csv")],
        "dev_a4_vive__fourth": [Path("a4/trial_1.csv")],
        "dev_b1_vive__excluded": [Path("b1/trial_1.csv")],
        "dev_mix1_vive__excluded": [Path("mix1/trial_1.csv")],
    }
    selected = select_sessions(
        sessions, ["dev_a1", "dev_a2", "dev_a3", "dev_a4"]
    )
    assert set(selected) == {
        "dev_a1_vive__first",
        "dev_a2_vive__second",
        "dev_a3_vive__third",
        "dev_a4_vive__fourth",
    }


def test_session_selection_finds_nested_export_ancestor_and_regroups() -> None:
    sessions = {
        "recordings": [
            Path("dataset/dev_a1_vive__first/recordings/trial_1.csv"),
            Path("dataset/dev_a2_vive__second/recordings/trial_2.csv"),
            Path("dataset/dev_a3_vive__third/recordings/trial_3.csv"),
            Path("dataset/dev_a4_vive__fourth/recordings/trial_4.csv"),
            Path("dataset/dev_b1_vive__excluded/recordings/trial_5.csv"),
        ]
    }
    selected = select_sessions(
        sessions, ["dev_a1", "dev_a2", "dev_a3", "dev_a4"]
    )
    assert set(selected) == {
        "dev_a1_vive__first",
        "dev_a2_vive__second",
        "dev_a3_vive__third",
        "dev_a4_vive__fourth",
    }
    assert sum(map(len, selected.values())) == 4
