from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np
import torch

from emg_touch.config import load_config
from emg_touch.data.tracked_dataset import emg_feature_count, imu_feature_count
from emg_touch.live_distillation import LiveDistillationModel, checkpoint_kind
from emg_touch.models.task_separated_complete_reach import (
    TaskSeparatedCompleteReachModel,
)
from scripts.train_complete_reach_model import make_complete_reach_window
from scripts.train_task_separated_complete_reach import (
    student_objective,
    task_separation_losses,
)


def _config() -> dict:
    return load_config("configs/tracked_task_separated_complete_reach.yaml")


def _window(config: dict, batch_size: int = 2) -> dict:
    length = 64
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    time = torch.linspace(0.0, 1.0, length)
    position = torch.stack([0.4 * time, -0.2 * time, 0.1 * time], dim=-1)
    velocity = torch.zeros_like(position)
    velocity[1:] = position[1:] - position[:-1]
    batch = {
        "emg": torch.randn(batch_size, length, emg_channels),
        "imu": torch.randn(batch_size, length, imu_channels),
        "position": position.unsqueeze(0).repeat(batch_size, 1, 1),
        "velocity": velocity.unsqueeze(0).repeat(batch_size, 1, 1),
        "lengths": torch.full((batch_size,), length),
        "onset": torch.full((batch_size,), 8),
        "screen_target": torch.tensor([[0.25, 0.75]]).repeat(batch_size, 1),
        "canvas": torch.tensor([[1920.0, 1080.0]]).repeat(batch_size, 1),
        "session": torch.zeros(batch_size, dtype=torch.long),
        "paths": [f"trial_{index:03d}.csv" for index in range(batch_size)],
    }
    return make_complete_reach_window(
        batch, 128, 16, int(config["model"]["teacher_trajectory_steps"]),
        np.random.default_rng(0), float(config["model"]["trajectory_limit_m"]),
        float(config["model"]["velocity_scale_mps"]), fixed_lead=0,
    )


def test_model_is_wearable_only_and_owns_two_heads() -> None:
    config = _config()
    model = TaskSeparatedCompleteReachModel(
        config, emg_feature_count(config["data"]), imu_feature_count(config["data"])
    ).eval()
    parameters = set(inspect.signature(model.student_forward).parameters)
    assert not parameters.intersection({
        "position", "velocity", "vive", "target", "lead_samples"
    })
    window = _window(config)
    with torch.no_grad():
        outputs = model.student_forward(
            window["emg"], window["imu"], window["time_mask"]
        )
    assert outputs["prediction"].shape == (2, 2)
    assert outputs["imu_base_trajectory"].shape == outputs["trajectory"].shape
    torch.testing.assert_close(outputs["trajectory"][:, 0], torch.zeros(2, 3))
    torch.testing.assert_close(outputs["trajectory"][:, -1], outputs["endpoint_3d"])
    assert checkpoint_kind(model.state_dict()) == "task_separated_complete_reach"


def test_3d_loss_cannot_rewrite_screen_semantic_adapter() -> None:
    config = _config()
    model = TaskSeparatedCompleteReachModel(
        config, emg_feature_count(config["data"]), imu_feature_count(config["data"])
    ).train()
    window = _window(config)
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"]
    )
    outputs["trajectory"].square().sum().backward()
    decoder = model.student.endpoint_decoder
    assert decoder.correction_adapter[1].weight.grad is not None
    assert decoder.screen_semantic[1].weight.grad is None


def test_objective_updates_screen_imu_base_and_bounded_correction() -> None:
    config = _config()
    model = TaskSeparatedCompleteReachModel(
        config, emg_feature_count(config["data"]), imu_feature_count(config["data"])
    ).train()
    window = _window(config)
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"], include_emg_only=True
    )
    teacher = model.teacher_forward(window["teacher_features"], sample=False)
    losses = student_objective(outputs, teacher, window, config)
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(task_separation_losses(outputs, window, config)[
        "task_base_preservation"
    ])
    losses["loss"].backward()
    decoder = model.student.endpoint_decoder
    assert decoder.point_head.direct.weight.grad is not None
    assert decoder.path_correction_head.weight.grad is not None
    assert decoder.endpoint_correction_head.weight.grad is not None
    assert model.student.imu_motion_head[-1].weight.grad is not None


def test_checkpoint_runs_in_live_api() -> None:
    config = _config()
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = TaskSeparatedCompleteReachModel(
        config, emg_channels, imu_channels
    ).eval()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "task_separated.pt"
        torch.save({"model_state": model.state_dict(), "config": config}, path)
        live = LiveDistillationModel("task-separated", path, device="cpu")
        samples = int(config["model"]["patch_length"]) + 4
        result = live.predict(
            np.zeros((samples, emg_channels), dtype=np.float32),
            np.zeros((samples, imu_channels), dtype=np.float32),
            canvas=(1920.0, 1080.0), effective_rate_hz=125.94,
        )
    assert result["kind"] == "task_separated_complete_reach"
    assert len(result["endpoint_3d_relative_m"]) == 3
    assert len(result["complete_trajectory_relative_m"]) == int(
        config["model"]["teacher_trajectory_steps"]
    )
