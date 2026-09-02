from __future__ import annotations

import inspect

import numpy as np
import torch
from torch import nn

from scripts.train_pointing_vae_model import evaluate, make_pointing_window
from emg_touch.models.grid_reach import PointingBottleneckModel


def _batch() -> dict[str, torch.Tensor]:
    steps = 80
    time = torch.arange(steps, dtype=torch.float32).view(1, steps, 1)
    return {
        "emg": time.expand(2, -1, 4).clone(),
        "imu": (10.0 * time).expand(2, -1, 24).clone(),
        "lengths": torch.tensor([steps, steps]),
        "onset": torch.tensor([20, 35]),
        "screen_target": torch.tensor([[0.2, 0.3], [0.7, 0.8]]),
        "canvas": torch.tensor([[1000.0, 500.0], [1000.0, 500.0]]),
    }


def test_onset_cutoff_is_independent_of_fixed_history() -> None:
    window = make_pointing_window(
        _batch(), minimum_prefix=16, patch_length=8,
        generator=np.random.default_rng(0), context_samples=40,
        cutoff_offsets=(-5,),
    )
    assert window is not None
    assert window["emg"].shape == (2, 40, 4)
    assert window["samples_past_onset"].tolist() == [-5, -5]
    # Cutoffs are 15 and 30. The history is causal and left-padded, so the
    # final observed sample is 14/29 and the mask contains 15/30 real samples.
    assert window["emg"][:, -1, 0].tolist() == [14.0, 29.0]
    assert window["time_mask"].sum(1).tolist() == [15, 30]


def test_future_samples_cannot_change_a_fixed_cutoff_window() -> None:
    original = _batch()
    changed = _batch()
    changed["emg"][:, 31:] += 10000.0
    changed["imu"][:, 31:] -= 10000.0
    kwargs = dict(
        minimum_prefix=16, patch_length=8, context_samples=40,
        cutoff_offsets=(-5,),
    )
    first = make_pointing_window(
        original, generator=np.random.default_rng(0), **kwargs
    )
    second = make_pointing_window(
        changed, generator=np.random.default_rng(0), **kwargs
    )
    assert first is not None and second is not None
    torch.testing.assert_close(first["emg"], second["emg"])
    torch.testing.assert_close(first["imu"], second["imu"])


class _ToyModel(nn.Module):
    use_imu = True

    def forward(self, emg, imu, time_mask):
        del time_mask
        value = (emg[:, -1, 0] + imu[:, -1, 0]) / 100.0
        prediction = torch.sigmoid(torch.stack([value, value], dim=-1))
        return {"prediction": prediction}


def test_evaluation_visits_each_offset_and_reports_paired_interventions() -> None:
    scores = evaluate(
        _ToyModel(), [_batch()], {"evaluation": {"paired_interventions": True}},
        torch.device("cpu"), minimum_prefix=16, patch_length=8, ablate=(),
        canvas_tensor=None, mean_target=torch.tensor([0.5, 0.5]),
        context_samples=40, cutoff_offsets=(-5, 0),
    )
    for offset in (-5, 0):
        assert f"direct_offset_{offset:+d}_px" in scores
    assert "without_emg_px" in scores
    assert "shuffled_emg_px" in scores
    assert "without_imu_px" in scores


def test_pointing_encoder_signature_has_no_tracker_inputs() -> None:
    parameters = set(inspect.signature(PointingBottleneckModel.forward).parameters)
    assert not ({"position", "velocity", "acceleration"} & parameters)
