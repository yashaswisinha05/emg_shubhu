from __future__ import annotations

import inspect

import torch

from emg_touch.config import load_config
from emg_touch.data.tracked_dataset import emg_feature_count, imu_feature_count
from emg_touch.live_distillation import checkpoint_kind
from emg_touch.models.rolling_dual_head_distillation import (
    RollingDualHeadDecoder,
    RollingDualHeadDistillationModel,
)
from scripts.train_rolling_dual_head_model import student_objective


def _config() -> dict:
    return load_config("configs/tracked_rolling_dual_head.yaml")


def test_dual_decoder_has_task_separated_gradients() -> None:
    config = _config()
    decoder = RollingDualHeadDecoder(config)
    latent_dim = int(config["model"]["latent_dim"])
    factor = torch.randn(3, latent_dim)
    bridge = torch.randn(3, latent_dim)
    horizon = torch.tensor([75.0, 200.0, 350.0])

    screen = decoder(factor, bridge, horizon)
    screen["prediction"].sum().backward()
    assert decoder.screen_semantic[1].weight.grad is not None
    assert decoder.motion_semantic[1].weight.grad is None

    decoder.zero_grad(set_to_none=True)
    motion = decoder(factor, bridge, horizon)
    motion["trajectory"].sum().backward()
    assert decoder.motion_semantic[1].weight.grad is not None
    assert decoder.screen_semantic[1].weight.grad is None


def test_rolling_model_forward_is_wearable_only_and_has_both_heads() -> None:
    config = _config()
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = RollingDualHeadDistillationModel(
        config, emg_channels, imu_channels
    ).eval()
    parameters = set(inspect.signature(model.student_forward).parameters)
    assert not parameters.intersection(
        {"position", "velocity", "vive", "trajectory_target", "lead_samples"}
    )

    batch, samples = 2, 64
    mask = torch.ones(batch, samples, dtype=torch.bool)
    with torch.no_grad():
        outputs = model.student_forward(
            torch.randn(batch, samples, emg_channels),
            torch.randn(batch, samples, imu_channels),
            mask,
            sample=False,
            include_emg_only=True,
        )
    steps = int(config["model"]["teacher_trajectory_steps"])
    assert outputs["prediction"].shape == (batch, 2)
    assert outputs["trajectory"].shape == (batch, steps, 3)
    assert outputs["guidance"]["horizon_expected_ms"].shape == (batch,)
    assert outputs["emg_only"]["prediction"].shape == (batch, 2)
    assert checkpoint_kind(model.state_dict()) == "rolling_dual_head"


def test_student_heads_can_copy_compatible_teacher_outputs() -> None:
    config = _config()
    model = RollingDualHeadDistillationModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    )
    model.initialise_student_decoder_from_teacher()
    student = model.student.endpoint_decoder
    torch.testing.assert_close(
        student.point_head.direct.weight,
        model.decoder.point_head.direct.weight,
    )
    torch.testing.assert_close(
        student.trajectory_head.weight,
        model.decoder.trajectory_head.weight,
    )
    torch.testing.assert_close(
        student.screen_shared[1].weight,
        model.decoder.trunk[1].weight,
    )
    torch.testing.assert_close(
        student.motion_shared[1].weight,
        model.decoder.trunk[1].weight,
    )


def test_combined_dual_head_training_loss_updates_both_heads() -> None:
    config = _config()
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = RollingDualHeadDistillationModel(
        config, emg_channels, imu_channels
    ).train()
    batch, samples = 2, 64
    steps = int(config["model"]["teacher_trajectory_steps"])
    mask = torch.ones(batch, samples, dtype=torch.bool)
    outputs = model.student_forward(
        torch.randn(batch, samples, emg_channels),
        torch.randn(batch, samples, imu_channels),
        mask,
        sample=False,
        include_emg_only=True,
    )
    teacher = model.teacher_forward(
        torch.randn(batch, steps, 6), sample=False
    )
    window = {
        "target": torch.rand(batch, 2),
        "canvas_size": torch.tensor([[1920.0, 1080.0]]).expand(batch, -1),
        "trajectory_target": 0.15 * torch.randn(batch, steps, 3),
        "lead_samples": torch.tensor([13, 38]),
        "time_mask": mask,
        "loss_weight": torch.ones(batch),
    }
    losses = student_objective(outputs, teacher, window, config)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    decoder = model.student.endpoint_decoder
    assert decoder.point_head.direct.weight.grad is not None
    assert decoder.trajectory_head.weight.grad is not None
    assert torch.isfinite(decoder.point_head.direct.weight.grad).all()
    assert torch.isfinite(decoder.trajectory_head.weight.grad).all()
