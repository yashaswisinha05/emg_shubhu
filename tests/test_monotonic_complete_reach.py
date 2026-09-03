from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np
import torch

from emg_touch.config import load_config
from emg_touch.data.tracked_dataset import emg_feature_count, imu_feature_count
from emg_touch.live_distillation import LiveDistillationModel, checkpoint_kind
from emg_touch.models.monotonic_complete_reach import (
    MonotonicCompleteReachModel,
)
from scripts.train_complete_reach_model import make_complete_reach_window
from scripts.train_monotonic_complete_reach import (
    monotonicity_statistics,
    student_objective,
)


def _config() -> dict:
    return load_config("configs/tracked_monotonic_complete_reach.yaml")


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


def _force_xyz_signs(model: MonotonicCompleteReachModel) -> None:
    """Select +x, -y, +z independently of the random test context."""
    head = model.student.endpoint_decoder.axis_direction_head
    with torch.no_grad():
        head.weight.zero_()
        model.student.endpoint_decoder.axis_direction_screen_head.weight.zero_()
        model.student.endpoint_decoder.axis_direction_screen_head.bias.zero_()
        head.bias.copy_(torch.tensor([
            -3.0, -3.0, 3.0,  # x positive
            3.0, -3.0, -3.0,  # y negative
            -3.0, -3.0, 3.0,  # z positive
        ]))


def test_forward_is_hard_monotonic_and_endpoint_exact() -> None:
    config = _config()
    model = MonotonicCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    ).eval()
    _force_xyz_signs(model)
    # Deliberately make the learned timing profile very non-uniform. Positive
    # cumulative increments must still prevent every reversal.
    with torch.no_grad():
        bias = model.student.endpoint_decoder.progress_increment_head.bias
        bias.copy_(torch.linspace(-8.0, 8.0, bias.numel()))
    window = _window(config)
    with torch.no_grad():
        outputs = model.student_forward(
            window["emg"], window["imu"], window["time_mask"]
        )
    trajectory = outputs["complete_trajectory"]
    torch.testing.assert_close(trajectory[:, 0], torch.zeros_like(trajectory[:, 0]))
    torch.testing.assert_close(trajectory[:, -1], outputs["endpoint_3d"])
    signed_steps = torch.diff(trajectory, dim=1) * outputs[
        "endpoint_3d"
    ][:, None, :]
    assert bool((signed_steps >= -1e-8).all())
    expected_signs = torch.tensor([1.0, -1.0, 1.0])
    assert (outputs["axis_direction_signs"][0] == expected_signs).all()
    statistics = monotonicity_statistics(trajectory)
    assert statistics["monotonic_any_reverse"].sum().item() == 0.0
    assert statistics["monotonic_reverse_step_fraction"].sum().item() == 0.0


def test_model_remains_wearable_only_and_has_distinct_checkpoint_kind() -> None:
    config = _config()
    model = MonotonicCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    )
    parameters = set(inspect.signature(model.student_forward).parameters)
    assert not parameters.intersection({
        "position", "velocity", "vive", "target", "lead_samples"
    })
    assert checkpoint_kind(model.state_dict()) == "monotonic_complete_reach"


def test_objective_updates_sign_magnitude_progress_and_emg_route() -> None:
    config = _config()
    model = MonotonicCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    ).train()
    _force_xyz_signs(model)
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
    decoder = model.student.endpoint_decoder
    for parameter in (
        decoder.axis_direction_head.weight,
        decoder.axis_direction_screen_head.weight,
        decoder.endpoint_3d_head.weight,
        decoder.progress_increment_head.weight,
        decoder.intent_to_motion[1].weight,
        decoder.screen_semantic[1].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_checkpoint_runs_in_live_and_manipulator_api_payload() -> None:
    config = _config()
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = MonotonicCompleteReachModel(
        config, emg_channels, imu_channels
    ).eval()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "monotonic.pt"
        torch.save({"model_state": model.state_dict(), "config": config}, path)
        live = LiveDistillationModel("monotonic", path, device="cpu")
        samples = int(config["model"]["patch_length"]) + 4
        result = live.predict(
            np.zeros((samples, emg_channels), dtype=np.float32),
            np.zeros((samples, imu_channels), dtype=np.float32),
            canvas=(1920.0, 1080.0),
            effective_rate_hz=125.94,
        )
    assert result["kind"] == "monotonic_complete_reach"
    assert set(result["axis_directions"]) == {"x", "y", "z"}
    trajectory = np.asarray(result["complete_trajectory_relative_m"])
    endpoint = np.asarray(result["endpoint_3d_relative_m"])
    np.testing.assert_allclose(trajectory[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(trajectory[-1], endpoint, atol=1e-7)
    assert np.all(np.diff(trajectory, axis=0) * endpoint[None, :] >= -1e-8)
