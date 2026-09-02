from __future__ import annotations

import numpy as np
import torch

from scripts.train_trajectory_model import TrajectoryModel, make_window
from emg_touch.data.tracked_dataset import emg_feature_bank, emg_feature_count
from emg_touch.models.trajectory_intent_vae import trajectory_loss


def _config(*, imu_dropout: float = 0.0) -> dict:
    return {
        "data": {
            "sensors": ["S0", "S4", "S8", "S12"],
            "sample_rate_hz": 100.0,
        },
        "model": {
            "d_model": 8,
            "separate_modality_encoders": True,
            "imu_modality_dropout": imu_dropout,
        },
        "virtual_leader": {
            "decimation": 1,
            "horizon": 4,
            "position_dim": 3,
            "substeps": 2,
            "prior_displacement_limit_m": 0.8,
        },
        "loss": {
            "trajectory_epsilon_m": 0.002,
            "trajectory_prior_sigma": 0.15,
            "kl_weight": 0.01,
        },
    }


def _batch() -> dict[str, torch.Tensor]:
    steps = 80
    time = torch.arange(steps, dtype=torch.float32).view(1, steps, 1)
    position = time.expand(2, -1, 3).clone()
    velocity = torch.ones_like(position)
    return {
        "emg": torch.randn(2, steps, 4),
        "imu": torch.randn(2, steps, 24),
        "position": position,
        "velocity": velocity,
        "position_mask": torch.ones(2, steps, dtype=torch.bool),
        "lengths": torch.tensor([steps, steps]),
        "onset": torch.tensor([20, 35]),
    }


def test_cutoff_offsets_are_independent_of_history_and_row_specific() -> None:
    made = make_window(
        _batch(), horizon=5, minimum_prefix=4,
        generator=np.random.default_rng(0), ablate=("position", "velocity"),
        relative=True, cutoff_offsets=(-5,),
    )
    assert made is not None
    window, _, _ = made
    assert window["samples_past_onset"].tolist() == [-5.0, -5.0]
    # Cutoffs are 15 and 30, so the last observed position is 14 and 29.
    assert window["_true_position"][:, -1, 0].tolist() == [14.0, 29.0]
    assert torch.count_nonzero(window["position"]) == 0
    assert torch.count_nonzero(window["velocity"]) == 0


def test_high_rate_emg_features_are_causal_and_have_expected_width() -> None:
    config = {
        "sensors": ["S0", "S4", "S8", "S12"],
        "emg_feature_windows_ms": [10.0, 25.0, 50.0],
        "emg_feature_kinds": [
            "rms", "waveform_length", "log_energy", "derivative"
        ],
    }
    raw = np.random.default_rng(2).normal(size=(100, 4)).astype(np.float32)
    changed_future = raw.copy()
    changed_future[80:] += 100.0
    original = emg_feature_bank(raw, 1000.0, config)
    changed = emg_feature_bank(changed_future, 1000.0, config)
    assert original.shape == (100, 48)
    assert emg_feature_count(config) == 48
    np.testing.assert_allclose(original[:80], changed[:80], rtol=0.0, atol=0.0)
    assert np.isfinite(original).all()


def test_wearable_prediction_is_invariant_to_tracker_values() -> None:
    torch.manual_seed(3)
    model = TrajectoryModel(_config(), task="wearable").eval()
    window = {
        "emg": torch.randn(2, 12, 4),
        "imu": torch.randn(2, 12, 24),
        "position": torch.zeros(2, 12, 3),
        "velocity": torch.zeros(2, 12, 3),
        "acceleration": torch.randn(2, 3),
    }
    changed = {
        **window,
        "position": torch.randn(2, 12, 3) * 1000.0,
        "velocity": torch.randn(2, 12, 3) * 1000.0,
        "acceleration": torch.randn(2, 3) * 1000.0,
    }
    with torch.no_grad():
        expected = model(window, 4)["trajectory"]
        actual = model(changed, 4)["trajectory"]
    torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)


def test_emg_only_objective_backpropagates_into_emg_encoder() -> None:
    torch.manual_seed(4)
    model = TrajectoryModel(_config(), task="wearable").train()
    window = {
        "emg": torch.randn(2, 12, 4),
        "imu": torch.randn(2, 12, 24),
        "position": torch.zeros(2, 12, 3),
        "velocity": torch.zeros(2, 12, 3),
        "acceleration": torch.zeros(2, 3),
    }
    target = torch.randn(2, 4, 3) * 0.05
    mask = torch.ones(2, 4, dtype=torch.bool)
    outputs = model(window, 4, include_emg_only=True)
    loss = trajectory_loss(outputs["emg_only_outputs"], target, mask, _config())["loss"]
    loss.backward()
    gradient = model.encoder.emg_project.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0


def test_full_imu_dropout_removes_imu_dependence_during_training() -> None:
    torch.manual_seed(5)
    encoder = TrajectoryModel(_config(imu_dropout=1.0), task="wearable").encoder.train()
    emg = torch.randn(2, 12, 4)
    zeros = torch.zeros(2, 12, 3)
    first = encoder(emg, torch.randn(2, 12, 24), zeros, zeros)
    second = encoder(emg, torch.randn(2, 12, 24) * 100.0, zeros, zeros)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
