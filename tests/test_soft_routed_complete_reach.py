from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np
import torch

from emg_touch.config import load_config
from emg_touch.data.tracked_dataset import emg_feature_count, imu_feature_count
from emg_touch.live_distillation import LiveDistillationModel, checkpoint_kind
from emg_touch.models.soft_routed_complete_reach import (
    SoftRoutedCompleteReachModel,
    scale_gradient,
)
from scripts.train_soft_routed_complete_reach import student_objective
from tests.test_task_separated_complete_reach import _window


def _config() -> dict:
    return load_config("configs/tracked_soft_routed_complete_reach.yaml")


def _model(config: dict) -> SoftRoutedCompleteReachModel:
    return SoftRoutedCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    )


def test_scale_gradient_preserves_forward_and_scales_backward() -> None:
    value = torch.tensor([2.0, -3.0], requires_grad=True)
    routed = scale_gradient(value, 0.10)
    torch.testing.assert_close(routed, value)
    routed.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, 0.10))


def test_model_is_wearable_only_and_predicts_both_tasks() -> None:
    config = _config()
    model = _model(config).eval()
    parameters = set(inspect.signature(model.student_forward).parameters)
    assert not parameters.intersection({
        "position", "velocity", "vive", "target", "lead_samples"
    })
    window = _window(config)
    with torch.no_grad():
        outputs = model.student_forward(
            window["emg"],
            window["imu"],
            window["time_mask"],
            include_emg_only=True,
        )
    assert outputs["prediction"].shape == (2, 2)
    assert outputs["trajectory"].shape == (2, 16, 3)
    assert outputs["axis_direction_logits"].shape == (2, 3, 3)
    torch.testing.assert_close(outputs["trajectory"][:, 0], torch.zeros(2, 3))
    torch.testing.assert_close(outputs["trajectory"][:, -1], outputs["endpoint_3d"])
    assert outputs["emg_only"]["trajectory"].shape == (2, 16, 3)
    assert checkpoint_kind(model.state_dict()) == "soft_routed_complete_reach"


def test_joint_objective_updates_screen_path_and_direction_heads() -> None:
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
    heads = model.student.soft_routed_reach_heads
    assert model.student.endpoint_decoder.point_head.direct.weight.grad is not None
    assert heads.path_correction_head.weight.grad is not None
    assert heads.endpoint_correction_head.weight.grad is not None
    assert heads.axis_direction_head.weight.grad is not None


def test_checkpoint_runs_in_live_api() -> None:
    config = _config()
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = _model(config).eval()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "soft_routed.pt"
        torch.save({"model_state": model.state_dict(), "config": config}, path)
        live = LiveDistillationModel("soft-routed", path, device="cpu")
        samples = int(config["model"]["patch_length"]) + 4
        result = live.predict(
            np.zeros((samples, emg_channels), dtype=np.float32),
            np.zeros((samples, imu_channels), dtype=np.float32),
            canvas=(1920.0, 1080.0),
            effective_rate_hz=125.94,
        )
    assert result["kind"] == "soft_routed_complete_reach"
    assert len(result["endpoint_3d_relative_m"]) == 3
    assert len(result["complete_trajectory_relative_m"]) == 16
