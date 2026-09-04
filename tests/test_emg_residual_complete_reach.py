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
from emg_touch.models.soft_routed_complete_reach import (
    SoftRoutedCompleteReachModel,
)
from scripts.train_emg_residual_complete_reach import student_objective
from tests.test_task_separated_complete_reach import _window


def _config() -> dict:
    return load_config("configs/tracked_emg_residual_complete_reach.yaml")


def _model(config: dict) -> EMGResidualCompleteReachModel:
    return EMGResidualCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    )


def test_soft_checkpoint_initializes_only_new_residual_parameters() -> None:
    config = _config()
    soft = SoftRoutedCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    )
    model = _model(config)
    missing, unexpected = model.load_state_dict(soft.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(
        key.startswith("student.emg_temporal_residual_head.") for key in missing
    )


def test_zero_initialized_residual_exactly_preserves_loaded_base() -> None:
    config = _config()
    model = _model(config).eval()
    window = _window(config)
    with torch.no_grad():
        outputs = model.student_forward(
            window["emg"], window["imu"], window["time_mask"]
        )
    torch.testing.assert_close(
        outputs["trajectory"], outputs["pre_emg_residual_trajectory"]
    )
    torch.testing.assert_close(
        outputs["endpoint_3d"], outputs["pre_emg_residual_endpoint"]
    )
    assert outputs["emg_temporal_attention"].shape[-2] == 16
    assert outputs["emg_temporal_attention"].shape[-1] > 0


def test_residual_objective_updates_temporal_emg_head() -> None:
    config = _config()
    model = _model(config).train()
    window = _window(config)
    outputs = model.student_forward(
        window["emg"],
        window["imu"],
        window["time_mask"],
        include_emg_only=True,
    )
    teacher = model.teacher_forward(window["teacher_features"], sample=False)
    losses = student_objective(outputs, teacher, window, config)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    head = model.student.emg_temporal_residual_head
    assert head.path_residual_head.weight.grad is not None
    assert bool(torch.count_nonzero(head.path_residual_head.weight.grad))
    assert head.endpoint_residual_head.weight.grad is not None


def test_checkpoint_runs_in_live_api() -> None:
    config = _config()
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = _model(config).eval()
    assert checkpoint_kind(model.state_dict()) == "emg_residual_complete_reach"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "emg_residual.pt"
        torch.save({"model_state": model.state_dict(), "config": config}, path)
        live = LiveDistillationModel("emg-residual", path, device="cpu")
        samples = int(config["model"]["patch_length"]) + 4
        result = live.predict(
            np.zeros((samples, emg_channels), dtype=np.float32),
            np.zeros((samples, imu_channels), dtype=np.float32),
            canvas=(1920.0, 1080.0),
            effective_rate_hz=125.94,
        )
    assert result["kind"] == "emg_residual_complete_reach"
    assert len(result["complete_trajectory_relative_m"]) == 16
