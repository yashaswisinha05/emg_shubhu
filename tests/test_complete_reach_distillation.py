from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np
import torch

from emg_touch.config import load_config
from emg_touch.data.tracked_dataset import emg_feature_count, imu_feature_count
from emg_touch.live_distillation import LiveDistillationModel, checkpoint_kind
from emg_touch.models.complete_reach_distillation import (
    CompleteReachDistillationModel,
)
from scripts.train_complete_reach_model import (
    make_complete_reach_window,
    milliseconds_to_samples,
    student_objective,
)


def _config() -> dict:
    return load_config("configs/tracked_complete_reach.yaml")


def _batch(config: dict, length: int = 64) -> dict:
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    time = torch.linspace(0.0, 1.0, length)
    position = torch.stack([0.4 * time, -0.2 * time, 0.1 * time], dim=-1)
    velocity = torch.zeros_like(position)
    velocity[1:] = position[1:] - position[:-1]
    return {
        "emg": torch.randn(1, length, emg_channels),
        "imu": torch.randn(1, length, imu_channels),
        "position": position.unsqueeze(0),
        "velocity": velocity.unsqueeze(0),
        "lengths": torch.tensor([length]),
        "onset": torch.tensor([8]),
        "screen_target": torch.tensor([[0.25, 0.75]]),
        "canvas": torch.tensor([[1920.0, 1080.0]]),
        "session": torch.tensor([0]),
        "paths": ["trial_001.csv"],
    }


def _window(config: dict, lead: int) -> dict:
    return make_complete_reach_window(
        _batch(config),
        context_samples=128,
        patch_length=16,
        teacher_steps=int(config["model"]["teacher_trajectory_steps"]),
        generator=np.random.default_rng(0),
        trajectory_limit_m=float(config["model"]["trajectory_limit_m"]),
        velocity_scale_mps=float(config["model"]["velocity_scale_mps"]),
        fixed_lead=lead,
    )


def test_zero_lead_is_real_and_includes_the_touch_sample() -> None:
    config = _config()
    assert milliseconds_to_samples(0.0, 125.94) == 0
    at_touch = _window(config, lead=0)
    earlier = _window(config, lead=10)
    assert at_touch["lead_samples"].item() == 0
    assert at_touch["time_mask"].sum().item() == 64
    assert earlier["time_mask"].sum().item() == 54


def test_complete_path_target_is_invariant_across_observation_times() -> None:
    config = _config()
    at_touch = _window(config, lead=0)
    earlier = _window(config, lead=20)
    torch.testing.assert_close(
        at_touch["trajectory_target"], earlier["trajectory_target"]
    )
    torch.testing.assert_close(
        at_touch["endpoint_3d_target"], earlier["endpoint_3d_target"]
    )
    torch.testing.assert_close(
        at_touch["trajectory_target"][:, 0], torch.zeros(1, 3)
    )
    torch.testing.assert_close(
        at_touch["trajectory_target"][:, -1],
        at_touch["endpoint_3d_target"],
    )


def test_deployment_forward_has_all_outputs_and_no_privileged_inputs() -> None:
    config = _config()
    model = CompleteReachDistillationModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    ).eval()
    parameters = set(inspect.signature(model.student_forward).parameters)
    assert not parameters.intersection({
        "position", "velocity", "vive", "target", "lead_samples"
    })
    window = _window(config, lead=0)
    with torch.no_grad():
        outputs = model.student_forward(
            window["emg"],
            window["imu"],
            window["time_mask"],
            include_emg_only=True,
        )
    steps = int(config["model"]["teacher_trajectory_steps"])
    assert outputs["prediction"].shape == (1, 2)
    assert outputs["endpoint_3d"].shape == (1, 3)
    assert outputs["complete_trajectory"].shape == (1, steps, 3)
    assert outputs["emg_only"]["endpoint_3d"].shape == (1, 3)
    assert checkpoint_kind(model.state_dict()) == "complete_reach"


def test_complete_reach_objective_updates_screen_path_and_endpoint() -> None:
    config = _config()
    model = CompleteReachDistillationModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    ).train()
    window = _window(config, lead=5)
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
    decoder = model.student.endpoint_decoder
    assert decoder.point_head.direct.weight.grad is not None
    assert decoder.trajectory_head.weight.grad is not None
    assert decoder.endpoint_3d_head.weight.grad is not None
    assert torch.isfinite(decoder.endpoint_3d_head.weight.grad).all()


def test_complete_reach_checkpoint_runs_through_live_api() -> None:
    config = _config()
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = CompleteReachDistillationModel(
        config, emg_channels, imu_channels
    ).eval()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "complete_reach.pt"
        torch.save(
            {"model_state": model.state_dict(), "config": config}, path
        )
        live = LiveDistillationModel("complete", path, device="cpu")
        samples = int(config["model"]["patch_length"]) + 4
        result = live.predict(
            np.zeros((samples, emg_channels), dtype=np.float32),
            np.zeros((samples, imu_channels), dtype=np.float32),
            canvas=(1920.0, 1080.0),
            effective_rate_hz=125.94,
        )
    assert result["kind"] == "complete_reach"
    assert len(result["endpoint_3d_relative_m"]) == 3
    assert len(result["complete_trajectory_relative_m"]) == int(
        config["model"]["teacher_trajectory_steps"]
    )
