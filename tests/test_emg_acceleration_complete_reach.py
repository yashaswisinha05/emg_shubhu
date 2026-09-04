from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np
import torch

from emg_touch.config import load_config
from emg_touch.data.tracked_dataset import emg_feature_count, imu_feature_count
from emg_touch.live_distillation import LiveDistillationModel, checkpoint_kind
from emg_touch.models.emg_acceleration_complete_reach import (
    EMGAccelerationCompleteReachModel,
)
from emg_touch.models.emg_residual_complete_reach import (
    EMGResidualCompleteReachModel,
)
from scripts.train_emg_acceleration_complete_reach import student_objective
from tests.test_task_separated_complete_reach import _window


def _config() -> dict:
    return load_config("configs/tracked_emg_acceleration_complete_reach.yaml")


def _model(config: dict) -> EMGAccelerationCompleteReachModel:
    return EMGAccelerationCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    )


def test_overlay_inherits_residual_configuration() -> None:
    config = _config()
    assert config["experiment_name"] == "tracked_emg_acceleration_complete_reach"
    assert config["data"]["include_session_prefixes"] == [
        "dev_a1", "dev_a2", "dev_a3", "dev_a4"
    ]
    assert config["model"]["emg_temporal_residual"]["warmup_epochs"] == 10
    assert config["model"]["emg_acceleration_dynamics"]["warmup_epochs"] == 10


def test_residual_checkpoint_is_preserved_before_dynamics_training() -> None:
    config = _config()
    previous = EMGResidualCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    ).eval()
    model = _model(config).eval()
    missing, unexpected = model.load_state_dict(previous.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(
        key.startswith("student.emg_acceleration_dynamics_head.") for key in missing
    )
    window = _window(config)
    with torch.no_grad():
        before = previous.student_forward(
            window["emg"], window["imu"], window["time_mask"]
        )
        after = model.student_forward(
            window["emg"], window["imu"], window["time_mask"]
        )
    torch.testing.assert_close(after["trajectory"], before["trajectory"])
    torch.testing.assert_close(after["endpoint_3d"], before["endpoint_3d"])
    torch.testing.assert_close(
        after["emg_integrated_position_residual"],
        torch.zeros_like(after["emg_integrated_position_residual"]),
    )


def test_acceleration_objective_updates_new_emg_head() -> None:
    config = _config()
    model = _model(config).train()
    window = _window(config)
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"], include_emg_only=True
    )
    teacher = model.teacher_forward(window["teacher_features"], sample=False)
    losses = student_objective(outputs, teacher, window, config)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    head = model.student.emg_acceleration_dynamics_head
    assert head.acceleration_head.weight.grad is not None
    assert bool(torch.count_nonzero(head.acceleration_head.weight.grad))
    assert head.duration_head[-1].weight.grad is not None


def test_deployable_signature_and_live_checkpoint_use_only_wearables() -> None:
    config = _config()
    model = _model(config).eval()
    parameters = set(inspect.signature(model.student_forward).parameters)
    assert not parameters.intersection({
        "position", "velocity", "acceleration", "vive", "target"
    })
    assert checkpoint_kind(model.state_dict()) == "emg_acceleration_complete_reach"
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "acceleration.pt"
        torch.save({
            "model_state": model.state_dict(), "config": config
        }, checkpoint)
        live = LiveDistillationModel("acceleration", checkpoint, device="cpu")
        samples = int(config["model"]["patch_length"]) + 4
        result = live.predict(
            np.zeros((samples, emg_channels), dtype=np.float32),
            np.zeros((samples, imu_channels), dtype=np.float32),
            canvas=(1920.0, 1080.0), effective_rate_hz=125.94,
        )
    assert result["kind"] == "emg_acceleration_complete_reach"
    assert len(result["complete_trajectory_relative_m"]) == 16
