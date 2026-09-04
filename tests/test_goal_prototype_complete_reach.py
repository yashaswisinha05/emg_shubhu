from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from emg_touch.config import load_config
from emg_touch.data.tracked_dataset import emg_feature_count, imu_feature_count
from emg_touch.live_distillation import LiveDistillationModel, checkpoint_kind
from emg_touch.models.emg_residual_complete_reach import (
    EMGResidualCompleteReachModel,
)
from emg_touch.models.goal_prototype_complete_reach import (
    GoalPrototypeCompleteReachModel,
)
from scripts.train_goal_prototype_complete_reach import student_objective
from tests.test_task_separated_complete_reach import _window


def _config() -> dict:
    return load_config("configs/tracked_goal_prototype_complete_reach.yaml")


def _model(config: dict) -> GoalPrototypeCompleteReachModel:
    return GoalPrototypeCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    )


def test_emg_residual_checkpoint_only_misses_goal_bridge() -> None:
    config = _config()
    previous = EMGResidualCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    )
    model = _model(config)
    missing, unexpected = model.load_state_dict(previous.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(
        key.startswith("student.goal_prototype_bridge.") for key in missing
    )


def test_zero_initialized_bridge_exactly_preserves_loaded_model() -> None:
    config = _config()
    model = _model(config).eval()
    window = _window(config)
    with torch.no_grad():
        outputs = model.student_forward(
            window["emg"], window["imu"], window["time_mask"]
        )
    torch.testing.assert_close(
        outputs["prediction"], outputs["pre_goal_prediction"]
    )
    torch.testing.assert_close(
        outputs["trajectory"], outputs["pre_goal_trajectory"]
    )
    torch.testing.assert_close(
        outputs["endpoint_3d"], outputs["pre_goal_endpoint"]
    )
    torch.testing.assert_close(
        outputs["trajectory"][:, -1], outputs["endpoint_3d"]
    )
    torch.testing.assert_close(
        outputs["goal_probabilities"].sum(dim=-1),
        torch.ones(outputs["prediction"].size(0)),
    )


def test_joint_objective_updates_prototypes_and_geometry() -> None:
    config = _config()
    model = _model(config).train()
    window = _window(config)
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"],
        include_emg_only=True,
    )
    teacher = model.teacher_forward(window["teacher_features"], sample=False)
    losses = student_objective(outputs, teacher, window, config)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    bridge = model.student.goal_prototype_bridge
    assert bridge.path_prototypes.grad is not None
    assert bool(torch.count_nonzero(bridge.path_prototypes.grad))
    assert bridge.endpoint_prototypes.grad is not None
    final = bridge.geometry_screen_correction[-1]
    assert final.weight.grad is not None
    assert bool(torch.count_nonzero(final.weight.grad))


def test_checkpoint_runs_in_live_api() -> None:
    config = _config()
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = _model(config).eval()
    assert checkpoint_kind(model.state_dict()) == "goal_prototype_complete_reach"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "goal.pt"
        torch.save({"model_state": model.state_dict(), "config": config}, path)
        live = LiveDistillationModel("goal", path, device="cpu")
        samples = int(config["model"]["patch_length"]) + 4
        result = live.predict(
            np.zeros((samples, emg_channels), dtype=np.float32),
            np.zeros((samples, imu_channels), dtype=np.float32),
            canvas=(1920.0, 1080.0),
            effective_rate_hz=125.94,
        )
    assert result["kind"] == "goal_prototype_complete_reach"
    assert len(result["complete_trajectory_relative_m"]) == 16
