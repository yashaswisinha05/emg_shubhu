from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from emg_touch.physics.manipulator_ik import ThreeRManipulator
from scripts.visualize_wearable_manipulator import collect_trials


class _WearableOnlyModel:
    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.calls = 0

    def student_forward(
        self,
        emg: torch.Tensor,
        imu: torch.Tensor,
        time_mask: torch.Tensor,
        sample: bool,
    ) -> dict[str, torch.Tensor]:
        self.calls += 1
        assert sample is False
        assert emg.ndim == 3 and imu.ndim == 3 and time_mask.ndim == 2
        # A stationary output makes it easy to prove that IK follows the model
        # rather than the non-stationary VIVE comparison path.
        return {
            "trajectory": torch.zeros(
                emg.size(0), self.steps, 3, dtype=emg.dtype, device=emg.device
            )
        }


def _runner_and_batch() -> tuple[SimpleNamespace, dict[str, object]]:
    steps = 4
    config = {
        "data": {"sample_rate_hz": 100.0, "decimation": 1},
        "model": {
            "teacher_trajectory_steps": steps,
            "trajectory_limit_m": 0.8,
            "velocity_scale_mps": 1.0,
        },
        "distillation": {"lead_window_ms": [50.0, 400.0]},
    }
    model = _WearableOnlyModel(steps)
    runner = SimpleNamespace(
        config=config,
        model=model,
        device=torch.device("cpu"),
        context_samples=20,
        patch_length=4,
    )
    samples = 101
    position = torch.zeros(1, samples, 3)
    position[0, :, 0] = torch.linspace(0.0, 0.5, samples)
    batch: dict[str, object] = {
        "emg": torch.randn(1, samples, 4),
        "imu": torch.randn(1, samples, 8),
        "position": position,
        "velocity": torch.zeros(1, samples, 3),
        "lengths": torch.tensor([samples]),
        "onset": torch.tensor([20]),
        "screen_target": torch.tensor([[100.0, 200.0]]),
        "canvas": torch.tensor([[1920.0, 1080.0]]),
        "paths": ["synthetic/trial_001.csv"],
    }
    return runner, batch


def test_full_reach_arm_follows_model_not_vive() -> None:
    runner, batch = _runner_and_batch()
    arm = ThreeRManipulator((0.5, 0.6))
    trials = collect_trials(
        runner,
        [batch],
        arm,
        count=1,
        lead_samples=20,
        initial_angles=np.deg2rad([0.0, 20.0, 90.0]),
        base_world=None,
        axis_order="xyz",
        axis_signs=(1.0, 1.0, 1.0),
        trajectory_window="full-reach",
    )
    trial = trials[0]
    assert runner.model.calls == 1
    assert trial["out_of_range"]
    assert np.isclose(trial["lead_ms"], 800.0)
    assert trial["cut_samples_past_onset"] == 0
    np.testing.assert_allclose(
        trial["followed"]["requested"], trial["predicted_path"]
    )
    assert not np.allclose(trial["predicted_path"], trial["vive_path"])


def test_fixed_horizon_remains_inside_training_range() -> None:
    runner, batch = _runner_and_batch()
    trials = collect_trials(
        runner,
        [batch],
        ThreeRManipulator((0.5, 0.6)),
        count=1,
        lead_samples=20,
        initial_angles=np.deg2rad([0.0, 20.0, 90.0]),
        base_world=None,
        axis_order="xyz",
        axis_signs=(1.0, 1.0, 1.0),
        trajectory_window="fixed-horizon",
    )
    trial = trials[0]
    assert not trial["out_of_range"]
    assert np.isclose(trial["lead_ms"], 200.0)
    assert trial["cut_samples_past_onset"] == 60
