from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from emg_touch.data.tracked_dataset import emg_feature_count, imu_feature_count
from emg_touch.live_distillation import LiveDistillationModel, checkpoint_kind
from emg_touch.models.deterministic_complete_reach import (
    DeterministicCompleteReachModel,
)
from scripts.train_deterministic_complete_reach import student_objective
from tests.test_task_separated_complete_reach import _config, _window


def _model(config: dict) -> DeterministicCompleteReachModel:
    return DeterministicCompleteReachModel(
        config, emg_feature_count(config["data"]), imu_feature_count(config["data"])
    )


def test_student_predictions_are_deterministic_and_bypass_teacher_bridge() -> None:
    config = _config()
    model = _model(config).eval()
    window = _window(config)
    with torch.no_grad():
        first = model.student_forward(
            window["emg"], window["imu"], window["time_mask"],
            sample=False, noise_scale=0.0,
        )
        for parameter in model.student.teacher_latent_bridge.parameters():
            parameter.add_(100.0 * torch.randn_like(parameter))
        for parameter in model.student.endpoint_decoder.parameters():
            parameter.add_(100.0 * torch.randn_like(parameter))
        second = model.student_forward(
            window["emg"], window["imu"], window["time_mask"],
            sample=True, noise_scale=100.0,
        )
    torch.testing.assert_close(first["prediction"], second["prediction"])
    torch.testing.assert_close(first["trajectory"], second["trajectory"])
    assert torch.count_nonzero(first["log_variance"]).item() == 0


def test_direct_heads_receive_screen_and_3d_gradients() -> None:
    config = _config()
    model = _model(config).train()
    window = _window(config)
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"], include_emg_only=True
    )
    teacher = model.teacher_forward(window["teacher_features"], sample=False)
    losses = student_objective(outputs, teacher, window, config)
    losses["loss"].backward()
    heads = model.student.deterministic_heads
    assert heads.point_head.direct.weight.grad is not None
    assert heads.path_correction_head.weight.grad is not None
    assert heads.endpoint_correction_head.weight.grad is not None
    assert model.student.imu_motion_head[-1].weight.grad is not None


def test_deterministic_checkpoint_runs_live() -> None:
    config = _config()
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = _model(config).eval()
    assert checkpoint_kind(model.state_dict()) == "deterministic_complete_reach"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "deterministic.pt"
        torch.save({"model_state": model.state_dict(), "config": config}, path)
        live = LiveDistillationModel("deterministic", path, device="cpu")
        count = int(config["model"]["patch_length"]) + 4
        result = live.predict(
            np.zeros((count, emg_channels), dtype=np.float32),
            np.zeros((count, imu_channels), dtype=np.float32),
            canvas=(1920.0, 1080.0), effective_rate_hz=125.94,
        )
    assert result["kind"] == "deterministic_complete_reach"
    assert len(result["complete_trajectory_relative_m"]) == int(
        config["model"]["teacher_trajectory_steps"]
    )
