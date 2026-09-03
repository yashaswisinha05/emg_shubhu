from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from emg_touch.physics.manipulator_ik import ThreeRManipulator
from scripts.visualize_complete_reach_manipulator import (
    collect_complete_reach_trials,
)


class _CompleteReachModel:
    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.valid_samples: list[int] = []

    def student_forward(
        self,
        emg: torch.Tensor,
        imu: torch.Tensor,
        time_mask: torch.Tensor,
        sample: bool,
    ) -> dict[str, torch.Tensor]:
        assert sample is False
        self.valid_samples.append(int(time_mask.sum()))
        batch = emg.size(0)
        return {
            # A stationary wearable prediction proves the arm does not follow
            # the deliberately non-stationary VIVE comparison trajectory.
            "complete_trajectory": torch.zeros(
                batch, self.steps, 3, dtype=emg.dtype
            ),
            "endpoint_3d": torch.zeros(batch, 3, dtype=emg.dtype),
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
        "distillation": {"lead_window_ms": [0.0, 400.0]},
    }
    model = _CompleteReachModel(steps)
    runner = SimpleNamespace(
        kind="complete_reach",
        config=config,
        model=model,
        device=torch.device("cpu"),
        context_samples=128,
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
        "screen_target": torch.tensor([[0.25, 0.75]]),
        "canvas": torch.tensor([[1920.0, 1080.0]]),
        "paths": ["synthetic/trial_001.csv"],
    }
    return runner, batch


def test_arm_follows_complete_model_path_at_touch_not_vive() -> None:
    runner, batch = _runner_and_batch()
    trials = collect_complete_reach_trials(
        runner,
        [batch],
        ThreeRManipulator((0.5, 0.6)),
        count=1,
        observation_lead_samples=0,
        initial_angles=np.deg2rad([0.0, 20.0, 90.0]),
        base_world=None,
        axis_order="xyz",
        axis_signs=(1.0, 1.0, 1.0),
    )
    trial = trials[0]
    assert runner.model.valid_samples == [101]
    assert trial["lead_ms"] == 0.0
    assert not trial["out_of_range"]
    np.testing.assert_allclose(
        trial["followed"]["requested"], trial["predicted_path"]
    )
    assert not np.allclose(trial["predicted_path"], trial["vive_path"])
    np.testing.assert_allclose(
        trial["explicit_endpoint"], trial["predicted_path"][-1]
    )


def test_earlier_observation_still_predicts_the_complete_reach() -> None:
    runner, batch = _runner_and_batch()
    trial = collect_complete_reach_trials(
        runner,
        [batch],
        ThreeRManipulator((0.5, 0.6)),
        count=1,
        observation_lead_samples=20,
        initial_angles=np.deg2rad([0.0, 20.0, 90.0]),
        base_world=None,
        axis_order="xyz",
        axis_signs=(1.0, 1.0, 1.0),
    )[0]
    assert runner.model.valid_samples == [81]
    assert np.isclose(trial["lead_ms"], 200.0)
    assert len(trial["predicted_path"]) == 5
    assert np.isclose(trial["time_ms"][-1], 800.0)
