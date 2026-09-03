from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np
import torch

from emg_touch.config import load_config
from emg_touch.data.tracked_dataset import emg_feature_count, imu_feature_count
from emg_touch.live_distillation import LiveDistillationModel, checkpoint_kind
from emg_touch.models.direction_aware_complete_reach import (
    DirectionAwareCompleteReachModel,
)
from scripts.train_complete_reach_model import make_complete_reach_window
from scripts.train_latent_distillation_model import student_validation_selection
from scripts.train_direction_aware_complete_reach import (
    axis_direction_labels,
    direction_losses,
    student_objective,
)


def _config() -> dict:
    return load_config("configs/tracked_direction_aware_complete_reach.yaml")


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
        batch,
        context_samples=128,
        patch_length=16,
        teacher_steps=int(config["model"]["teacher_trajectory_steps"]),
        generator=np.random.default_rng(0),
        trajectory_limit_m=float(config["model"]["trajectory_limit_m"]),
        velocity_scale_mps=float(config["model"]["velocity_scale_mps"]),
        fixed_lead=0,
    )


def test_axis_direction_labels_include_stationary_margin() -> None:
    values = torch.tensor([[-0.10, 0.005, 0.10]])
    labels = axis_direction_labels(values, stationary_threshold_m=0.015)
    assert labels.tolist() == [[0, 1, 2]]


def test_direction_experiment_selects_composite_without_overwriting_pixels() -> None:
    config = _config()
    validation = {"student_px": 180.0, "direction_selection_score": 225.0}
    metric, value = student_validation_selection(validation, config)
    assert metric == "direction_selection_score"
    assert value == 225.0
    assert validation["student_px"] == 180.0
    default_config = {"distillation": {}}
    assert student_validation_selection(validation, default_config) == (
        "student_px", 180.0
    )


def test_wrong_way_motion_has_larger_direction_loss() -> None:
    config = _config()
    target = torch.tensor([[0.4, -0.2, 0.1]])
    path = torch.stack([
        torch.zeros(1, 3), target * 0.5, target
    ], dim=1)
    window = {
        "endpoint_3d_target": target,
        "trajectory_target": path,
    }
    logits = torch.zeros(1, 3, 3)
    good = {
        "endpoint_3d": target.clone(),
        "trajectory": path.clone(),
        "axis_direction_logits": logits,
    }
    bad = {
        "endpoint_3d": -target,
        "trajectory": -path,
        "axis_direction_logits": logits,
    }
    good_losses = direction_losses(good, window, config)
    bad_losses = direction_losses(bad, window, config)
    assert bad_losses["direction_endpoint"] > good_losses["direction_endpoint"]
    assert bad_losses["direction_path"] > good_losses["direction_path"]
    assert bad_losses["direction_velocity"] > good_losses["direction_velocity"]


def test_direction_model_is_wearable_only_and_routes_intent_to_motion() -> None:
    config = _config()
    model = DirectionAwareCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    ).train()
    parameters = set(inspect.signature(model.student_forward).parameters)
    assert not parameters.intersection({
        "position", "velocity", "vive", "target", "lead_samples"
    })
    window = _window(config)
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"], include_emg_only=True
    )
    assert outputs["axis_direction_logits"].shape == (2, 3, 3)
    assert outputs["intent_to_motion_gate"].shape == (2,)
    outputs["trajectory"].sum().backward()
    decoder = model.student.endpoint_decoder
    assert decoder.intent_to_motion[1].weight.grad is not None
    assert decoder.screen_semantic[1].weight.grad is not None
    assert checkpoint_kind(model.state_dict()) == "direction_aware_complete_reach"


def test_combined_objective_updates_direction_and_trajectory_heads() -> None:
    config = _config()
    model = DirectionAwareCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    ).train()
    window = _window(config)
    outputs = model.student_forward(
        window["emg"], window["imu"], window["time_mask"], include_emg_only=True
    )
    teacher = model.teacher_forward(window["teacher_features"], sample=False)
    losses = student_objective(outputs, teacher, window, config)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    decoder = model.student.endpoint_decoder
    assert decoder.axis_direction_head.weight.grad is not None
    assert decoder.endpoint_3d_head.weight.grad is not None
    assert decoder.trajectory_head.weight.grad is not None


def test_direction_checkpoint_runs_in_live_api() -> None:
    config = _config()
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = DirectionAwareCompleteReachModel(
        config, emg_channels, imu_channels
    ).eval()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "direction.pt"
        torch.save({"model_state": model.state_dict(), "config": config}, path)
        live = LiveDistillationModel("direction", path, device="cpu")
        samples = int(config["model"]["patch_length"]) + 4
        result = live.predict(
            np.zeros((samples, emg_channels), dtype=np.float32),
            np.zeros((samples, imu_channels), dtype=np.float32),
            canvas=(1920.0, 1080.0),
            effective_rate_hz=125.94,
        )
    assert result["kind"] == "direction_aware_complete_reach"
    assert set(result["axis_directions"]) == {"x", "y", "z"}
    assert len(result["complete_trajectory_relative_m"]) == int(
        config["model"]["teacher_trajectory_steps"]
    )
