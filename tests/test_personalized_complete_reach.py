from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from emg_touch.config import load_config
from emg_touch.data.tracked_dataset import emg_feature_count, imu_feature_count
from emg_touch.live_distillation import LiveDistillationModel, checkpoint_kind
from emg_touch.models.emg_acceleration_complete_reach import (
    EMGAccelerationCompleteReachModel,
)
from emg_touch.models.personalized_complete_reach import (
    PersonalizedCompleteReachModel,
)
from scripts.train_personalized_complete_reach import student_objective
from scripts import train_personalized_complete_reach as personalization
from tests.test_task_separated_complete_reach import _window


def _config() -> dict:
    config = load_config("configs/tracked_personalized_complete_reach.yaml")
    config["virtual_leader"]["session_count"] = 4
    return config


def _model(config: dict) -> PersonalizedCompleteReachModel:
    return PersonalizedCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    )


def test_population_checkpoint_is_exactly_preserved_initially() -> None:
    config = _config()
    population = EMGAccelerationCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    ).eval()
    personalized = _model(config).eval()
    missing, unexpected = personalized.load_state_dict(
        population.state_dict(), strict=False
    )
    assert not unexpected
    assert missing
    assert all(key.startswith("student.candidate_personalization.") for key in missing)
    window = _window(config)
    with torch.no_grad():
        before = population.student_forward(
            window["emg"], window["imu"], window["time_mask"]
        )
        after = personalized.student_forward(
            window["emg"], window["imu"], window["time_mask"]
        )
    torch.testing.assert_close(after["prediction"], before["prediction"])
    torch.testing.assert_close(after["trajectory"], before["trajectory"])
    torch.testing.assert_close(after["endpoint_3d"], before["endpoint_3d"])


def test_personalization_loss_reaches_low_rank_adapter() -> None:
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
    head = model.student.candidate_personalization
    assert head.screen_head.weight.grad is not None
    assert bool(torch.count_nonzero(head.screen_head.weight.grad))
    assert head.path_head.weight.grad is not None
    assert bool(torch.count_nonzero(head.path_head.weight.grad))


def test_personalized_live_model_still_accepts_only_wearables() -> None:
    config = _config()
    model = _model(config).eval()
    parameters = set(inspect.signature(model.student_forward).parameters)
    assert not parameters.intersection({"position", "velocity", "vive", "target"})
    assert checkpoint_kind(model.state_dict()) == "personalized_complete_reach"
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "personalized.pt"
        torch.save({"model_state": model.state_dict(), "config": config}, path)
        live = LiveDistillationModel("candidate", path, device="cpu")
        samples = int(config["model"]["patch_length"]) + 4
        result = live.predict(
            np.zeros((samples, emg_channels), dtype=np.float32),
            np.zeros((samples, imu_channels), dtype=np.float32),
            canvas=(1920.0, 1080.0), effective_rate_hz=125.94,
        )
    assert result["kind"] == "personalized_complete_reach"
    assert len(result["complete_trajectory_relative_m"]) == 16


def test_multiple_candidates_are_split_and_normalized_separately() -> None:
    config = load_config("configs/tracked_soft_routed_complete_reach.yaml")
    prefixes = ["32e00ff16111", "shubhamcal1_b0f8c99b", "f5a69f99ddeb"]
    config["data"]["include_session_prefixes"] = prefixes
    discovered = {
        prefix: [
            Path("/data") / prefix / f"trial_{index:03d}.csv"
            for index in range(50)
        ]
        for prefix in prefixes
    }
    emg_dim = personalization.raw_emg_feature_count(config["data"])
    imu_dim = imu_feature_count(config["data"])
    with (
        patch.object(personalization, "discover_trials", return_value=discovered),
        patch.object(
            personalization,
            "session_emg_scale",
            side_effect=lambda trials, data: np.full(
                emg_dim, float(len(trials)), dtype=np.float32
            ),
        ) as emg_scale,
        patch.object(
            personalization,
            "session_imu_statistics",
            side_effect=lambda trials, data: (
                np.zeros(imu_dim, dtype=np.float32),
                np.ones(imu_dim, dtype=np.float32),
            ),
        ) as imu_statistics,
        patch.object(personalization, "fit_training_emg_pca"),
    ):
        train, validation, test = personalization.build_candidate_loaders(
            config, "/data", None
        )
    assert len(train.dataset) == 90
    assert len(validation.dataset) == 30
    assert len(test.dataset) == 30
    assert emg_scale.call_count == 3
    assert imu_statistics.call_count == 3
    assert config["virtual_leader"]["session_count"] == 3
    assert set(personalization._CANDIDATE_CALIBRATIONS) == set(prefixes)
