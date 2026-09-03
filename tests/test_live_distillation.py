from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.live_best_model_ui import StudentForwardAdapter, _session_owner
from scripts.live_inference import replay_trial
from scripts.run_live_distillation_ui import LiveApplication, parse_checkpoint
from tests.test_channel_horizon_distillation import _enhanced_config
from emg_touch.live_distillation import (
    LiveDistillationModel,
    LiveFeaturePipeline,
    checkpoint_kind,
)
from emg_touch.models.channel_horizon_distillation import (
    ChannelHorizonLatentDistillationModel,
)
from emg_touch.models.semantic_residual_distillation import (
    SemanticResidualDistillationModel,
)


def _write_live_files(directory: Path) -> tuple[Path, Path, dict]:
    config = _enhanced_config()
    config["model"]["context_ms"] = 240.0
    model = ChannelHorizonLatentDistillationModel(config, 4, 24)
    checkpoint = directory / "final.pt"
    torch.save(
        {"model_state": model.state_dict(), "config": config}, checkpoint
    )
    calibration = directory / "calibration.npz"
    np.savez_compressed(
        calibration,
        emg_scale=np.ones(4, dtype=np.float32),
        imu_center=np.zeros(24, dtype=np.float32),
        imu_scale=np.ones(24, dtype=np.float32),
    )
    return checkpoint, calibration, config


def _raw_rows(count: int, start: int = 0) -> tuple[list[float], list, list]:
    indices = np.arange(start, start + count, dtype=np.float32)
    times = (indices / 100.0).tolist()
    emg = np.stack([
        np.sin(indices / 7.0),
        np.cos(indices / 9.0),
        np.sin(indices / 11.0),
        np.cos(indices / 13.0),
    ], axis=1).tolist()
    imu = np.zeros((count, 24), dtype=np.float32)
    imu[:, 2::6] = 9.8
    imu[:, 0] = indices / 1000.0
    return times, emg, imu.tolist()


def test_checkpoint_detection_and_named_argument() -> None:
    state = {
        "teacher.weight": torch.zeros(1),
        "student.channel_gate.sensor_bias": torch.zeros(4),
    }
    assert checkpoint_kind(state) == "channel_horizon"
    name, path = parse_checkpoint("Best model=/tmp/final.pt")
    assert name == "Best model"
    assert path == Path("/tmp/final.pt")
    semantic = {
        **state,
        "student.fused_endpoint_residual.network.0.weight": torch.zeros(1),
    }
    assert checkpoint_kind(semantic) == "semantic_residual"


def test_semantic_residual_checkpoint_loads_in_live_runner() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        _, _, config = _write_live_files(directory)
        model = SemanticResidualDistillationModel(config, 4, 24)
        checkpoint = directory / "semantic.pt"
        torch.save(
            {"model_state": model.state_dict(), "config": config}, checkpoint
        )
        runner = LiveDistillationModel("Semantic residual", checkpoint, "cpu")
        assert runner.kind == "semantic_residual"


class _RecordingStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def student_forward(
        self, emg: torch.Tensor, imu: torch.Tensor, time_mask: torch.Tensor,
        sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        assert not sample
        self.seen.append((emg.clone(), imu.clone(), time_mask.clone()))
        return {"prediction": torch.full((emg.size(0), 2), 0.5)}


def test_best_ui_adapter_uses_fixed_causal_training_context() -> None:
    student = _RecordingStudent()
    adapter = StudentForwardAdapter(student, context_samples=24)
    records = replay_trial(
        adapter,
        None,
        torch.randn(40, 4),
        torch.randn(40, 6),
        onset=0,
        touch=39,
        minimum_prefix=4,
        patch_length=4,
        stride=5,
        canvas=torch.tensor([1000.0, 500.0]),
        target_px=torch.tensor([500.0, 250.0]),
        maximum_prefix=24,
    )
    assert records
    assert all(emg.shape == (1, 24, 4) for emg, _, _ in student.seen)
    first_emg, _, first_mask = student.seen[0]
    assert torch.count_nonzero(first_emg[:, :-4]) == 0
    assert first_mask.sum() == 4
    assert student.seen[-1][2].sum() == 24


def test_replay_can_delay_first_prediction_until_600ms_after_onset() -> None:
    student = _RecordingStudent()
    adapter = StudentForwardAdapter(student, context_samples=80)
    records = replay_trial(
        adapter,
        None,
        torch.randn(120, 4),
        torch.randn(120, 6),
        onset=10,
        touch=119,
        minimum_prefix=4,
        patch_length=4,
        stride=5,
        canvas=torch.tensor([1000.0, 500.0]),
        target_px=torch.tensor([500.0, 250.0]),
        maximum_prefix=80,
        prediction_delay_samples=60,
    )
    assert records[0]["sample"] == 70
    assert all(record["sample"] >= 70 for record in records)


def test_best_ui_regroups_nested_trials_by_selected_session() -> None:
    path = Path(
        "/dataset/dev_a2_vive__abc/nested/export/trial_001.csv"
    )
    assert _session_owner(path, ["dev_a1", "dev_a2"], "export") == (
        "dev_a2_vive__abc"
    )
    assert _session_owner(path, ["dev_a1"], "export") is None


def test_live_preprocessing_is_causal() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        _, calibration, config = _write_live_files(directory)
        first = LiveFeaturePipeline(config, calibration)
        second = LiveFeaturePipeline(config, calibration)
        common = _raw_rows(80)
        first.add_samples(*common)
        second.add_samples(*common)
        emg_before, imu_before, _ = first.processed()

        times, emg, imu = _raw_rows(40, start=80)
        emg = (1000.0 * np.asarray(emg)).tolist()
        imu = (1000.0 + np.asarray(imu)).tolist()
        second.add_samples(times, emg, imu)
        emg_after, imu_after, _ = second.processed()
        torch.testing.assert_close(
            torch.from_numpy(emg_before), torch.from_numpy(emg_after[:80])
        )
        torch.testing.assert_close(
            torch.from_numpy(imu_before), torch.from_numpy(imu_after[:80])
        )


def test_ground_truth_changes_errors_but_not_predictions() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        checkpoint, calibration, config = _write_live_files(directory)
        runner = LiveDistillationModel("Channel + horizon", checkpoint, "cpu")
        pipeline = LiveFeaturePipeline(config, calibration)
        application = LiveApplication(
            [runner], pipeline, interval_ms=40.0, canvas=(1000.0, 500.0)
        )
        times, emg, imu = _raw_rows(64)
        application.handle_event({
            "event": "samples", "time_s": times, "emg": emg, "imu": imu,
        })
        before = application.snapshot()
        assert before["predictions"]
        point_before = before["predictions"][-1]["models"][0]
        assert "error_px" not in point_before

        application.handle_event({
            "event": "target", "x_px": 700.0, "y_px": 180.0,
        })
        after = application.snapshot()
        point_after = after["predictions"][-1]["models"][0]
        assert point_after["x_px"] == point_before["x_px"]
        assert point_after["y_px"] == point_before["y_px"]
        assert point_after["error_px"] >= 0.0
        assert point_after["horizon_ms"] >= 0.0
        assert set(point_after["channel_attention"]) == {"S0", "S4", "S8", "S12"}


def test_live_ui_is_streaming_not_replay() -> None:
    html = (
        Path(__file__).resolve().parents[1]
        / "src/emg_touch/static/live_distillation.html"
    ).read_text(encoding="utf-8")
    lowered = html.lower()
    assert "/api/event" in html
    assert "live model outputs" in lowered
    assert "replay" not in lowered
