from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from emg_touch.data.tracked_dataset import emg_feature_count, imu_feature_count
from emg_touch.live_distillation import LiveDistillationModel, checkpoint_kind
from emg_touch.models.asymmetric_intent_motion import AsymmetricIntentMotionModel
from scripts.train_asymmetric_intent_motion import student_objective
from tests.test_task_separated_complete_reach import _config, _window


def _model(config: dict) -> AsymmetricIntentMotionModel:
    return AsymmetricIntentMotionModel(
        config, emg_feature_count(config["data"]), imu_feature_count(config["data"])
    )


def _has_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in module.parameters()
    )


def test_screen_gradient_is_owned_by_emg_not_imu_encoder() -> None:
    config = _config()
    model = _model(config).train()
    window = _window(config)
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"]
    )
    outputs["prediction"].sum().backward()
    assert _has_gradient(model.student.emg_encoder)
    assert not _has_gradient(model.student.imu_encoder)


def test_trajectory_gradient_is_owned_by_imu_not_emg_encoder() -> None:
    config = _config()
    model = _model(config).train()
    window = _window(config)
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"]
    )
    outputs["trajectory"].square().sum().backward()
    assert _has_gradient(model.student.imu_encoder)
    assert not _has_gradient(model.student.emg_encoder)


def test_joint_objective_updates_both_owned_routes() -> None:
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
    assert _has_gradient(model.student.emg_encoder)
    assert _has_gradient(model.student.imu_encoder)
    assert model.student.endpoint_decoder.point_head.direct.weight.grad is not None
    assert (
        model.student.asymmetric_motion_heads.path_correction_head.weight.grad
        is not None
    )


def test_asymmetric_checkpoint_runs_live() -> None:
    config = _config()
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = _model(config).eval()
    assert checkpoint_kind(model.state_dict()) == "asymmetric_intent_motion"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "asymmetric.pt"
        torch.save({"model_state": model.state_dict(), "config": config}, path)
        live = LiveDistillationModel("asymmetric", path, device="cpu")
        count = int(config["model"]["patch_length"]) + 4
        result = live.predict(
            np.zeros((count, emg_channels), dtype=np.float32),
            np.zeros((count, imu_channels), dtype=np.float32),
            canvas=(1920.0, 1080.0), effective_rate_hz=125.94,
        )
    assert result["kind"] == "asymmetric_intent_motion"
    assert len(result["endpoint_3d_relative_m"]) == 3
