#!/usr/bin/env python3
"""Parameter-efficient adaptation of the best model to one new candidate.

The candidate's trials are split before normalization. EMG/IMU statistics and
PCA are fitted on the training subset only. A zero-initialized low-rank adapter
first trains alone, then the existing output heads are unfrozen at low learning
rate. The wearable encoders remain frozen to avoid overfitting 120--160 trials.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import train_channel_horizon_distillation_model as channel  # noqa: E402
from scripts import train_complete_reach_model as complete  # noqa: E402
from scripts import train_emg_acceleration_complete_reach as acceleration  # noqa: E402
from scripts import train_latent_distillation_model as base  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    apply_sensor_local_pca,
    collate_tracked,
    discover_trials,
    emg_feature_count,
    fit_training_emg_pca,
    imu_feature_count,
    raw_emg_feature_count,
    session_emg_scale,
    session_imu_statistics,
    TrackedTrajectoryDataset,
)
from emg_touch.models.personalized_complete_reach import (  # noqa: E402
    PersonalizedCompleteReachModel,
)


_ORIGINAL_TRAIN_STUDENT_PHASE = base.train_student_phase
_ORIGINAL_SET_TRAINABLE = base.set_trainable
_INITIAL_CHECKPOINT: Path | None = None
_CANDIDATE_CALIBRATION: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def _matches_prefix(path: Path, prefixes: list[str]) -> bool:
    for part in path.parts:
        lowered = part.lower()
        for prefix in prefixes:
            if (
                lowered == prefix
                or lowered.startswith(prefix + "_")
                or lowered.startswith(prefix + "-")
            ):
                return True
    return False


class CandidateDataset(TrackedTrajectoryDataset):
    """Apply one training-only calibration to all candidate splits."""

    def __init__(
        self,
        trials: list[Path],
        data_config: dict[str, Any],
        cache_dir: Path | None,
        emg_scale: np.ndarray | None,
        imu_statistics: tuple[np.ndarray, np.ndarray] | None,
    ) -> None:
        super().__init__(
            trials, data_config, cache_dir, session_index={}, apply_emg_pca=False
        )
        self.candidate_emg_scale = emg_scale
        self.candidate_imu_statistics = imu_statistics

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = super().__getitem__(index)
        if result.get("unusable"):
            return result
        if self.candidate_emg_scale is not None:
            result["emg"] = result["emg"] / torch.from_numpy(
                self.candidate_emg_scale
            )
        result["emg"] = torch.from_numpy(
            apply_sensor_local_pca(result["emg"].numpy(), self.data_config)
        )
        if self.candidate_imu_statistics is not None:
            centre, spread = self.candidate_imu_statistics
            result["imu"] = (
                result["imu"] - torch.from_numpy(centre)
            ) / torch.from_numpy(spread)
        result["session"] = 0
        return result


def build_candidate_loaders(
    config: dict[str, Any], root: str | Path, cache_dir: str | Path | None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    global _CANDIDATE_CALIBRATION
    prefixes = [
        str(value).strip().lower()
        for value in config["data"].get("include_session_prefixes", [])
        if str(value).strip()
    ]
    if not prefixes:
        raise ValueError("provide exactly one candidate using --session-prefixes")
    trials = [
        path
        for paths in discover_trials(root).values()
        for path in paths
        if _matches_prefix(path, prefixes)
    ]
    if not trials:
        raise ValueError(
            "candidate prefix matched no trial_*.csv: " + ", ".join(prefixes)
        )
    generator = np.random.default_rng(int(config.get("seed", 42)))
    generator.shuffle(trials)
    validation_fraction = float(config["data"].get("validation_fraction", 0.2))
    test_fraction = float(config["data"].get("test_fraction", 0.2))
    n_test = max(1, int(round(len(trials) * test_fraction)))
    n_validation = max(1, int(round(len(trials) * validation_fraction)))
    train = trials[n_test + n_validation:]
    validation = trials[n_test:n_test + n_validation]
    test = trials[:n_test]
    if len(train) < 20:
        raise ValueError(
            f"only {len(train)} candidate training trials remain after splitting"
        )

    # Strict evaluation: neither validation nor test samples contribute to
    # normalization or PCA fitting.
    emg_scale = None
    if bool(config["data"].get("emg_session_normalise", True)):
        emg_scale = session_emg_scale(train, config["data"])
    imu_statistics = None
    if bool(config["data"].get("imu_session_normalise", True)):
        imu_statistics = session_imu_statistics(train, config["data"])
    trial_sessions = {str(path): "candidate" for path in trials}
    scales = {"candidate": emg_scale} if emg_scale is not None else {}
    fit_training_emg_pca(
        train, config["data"], scales, trial_sessions=trial_sessions
    )
    if emg_scale is None:
        emg_scale = np.ones(
            raw_emg_feature_count(config["data"]), dtype=np.float32
        )
    if imu_statistics is None:
        imu_dim = imu_feature_count(config["data"])
        imu_statistics = (
            np.zeros(imu_dim, dtype=np.float32),
            np.ones(imu_dim, dtype=np.float32),
        )
    _CANDIDATE_CALIBRATION = (
        np.asarray(emg_scale, dtype=np.float32),
        np.asarray(imu_statistics[0], dtype=np.float32),
        np.asarray(imu_statistics[1], dtype=np.float32),
    )
    config.setdefault("virtual_leader", {})["session_count"] = int(
        config["model"]["candidate_personalization"].get(
            "source_session_count", 4
        )
    )
    cache = Path(cache_dir) if cache_dir else None
    batch_size = int(config["training"].get("batch_size", 16))
    workers = int(config["training"].get("num_workers", 0))

    def loader(selected: list[Path], shuffle: bool) -> DataLoader:
        return DataLoader(
            CandidateDataset(
                selected, config["data"], cache, emg_scale, imu_statistics
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=collate_tracked,
            drop_last=False,
        )

    print(
        f"candidate {', '.join(prefixes)}: {len(train)} train | "
        f"{len(validation)} validation | {len(test)} untouched test"
    )
    print("normalization/PCA fit on candidate training trials only")
    return loader(train, True), loader(validation, False), loader(test, False)


def personalization_losses(
    outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    settings = config["model"]["candidate_personalization"]
    pixel_scale = max(float(settings.get("pixel_normalizer_px", 100.0)), 1e-6)
    spatial_scale = max(float(config["model"].get("trajectory_limit_m", 0.8)), 1e-6)
    screen_target = window["target"] - outputs[
        "pre_personalization_prediction"
    ].detach()
    screen = F.smooth_l1_loss(
        outputs["personalization_screen_residual"]
        * window["canvas_size"] / pixel_scale,
        screen_target * window["canvas_size"] / pixel_scale,
        beta=0.05,
    )
    path_target = window["trajectory_target"] - outputs[
        "pre_personalization_trajectory"
    ].detach()
    path = F.smooth_l1_loss(
        outputs["personalization_path_residual"] / spatial_scale,
        path_target / spatial_scale,
        beta=0.05,
    )
    endpoint_target = window["endpoint_3d_target"] - outputs[
        "pre_personalization_endpoint"
    ].detach()
    endpoint = F.smooth_l1_loss(
        outputs["personalization_endpoint_residual"] / spatial_scale,
        endpoint_target / spatial_scale,
        beta=0.05,
    )
    regularization = (
        outputs["personalization_screen_raw"].square().mean()
        + outputs["personalization_path_raw"].square().mean()
        + outputs["personalization_endpoint_raw"].square().mean()
    )
    return {
        "personalization_screen": screen,
        "personalization_path": path,
        "personalization_endpoint": endpoint,
        "personalization_regularization": regularization,
    }


def student_objective(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    combined = acceleration.student_objective(
        outputs, teacher_outputs, window, config
    )
    settings = config["model"]["candidate_personalization"]
    losses = personalization_losses(outputs, window, config)
    combined["loss"] = (
        combined["loss"]
        + float(settings.get("screen_weight", 0.75))
        * losses["personalization_screen"]
        + float(settings.get("path_weight", 0.50))
        * losses["personalization_path"]
        + float(settings.get("endpoint_weight", 0.50))
        * losses["personalization_endpoint"]
        + float(settings.get("regularization_weight", 0.01))
        * losses["personalization_regularization"]
    )
    combined.update({name: value.detach() for name, value in losses.items()})
    return combined


def load_initial_checkpoint(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    model = args[0] if args else kwargs["model"]
    output = Path(args[11] if len(args) > 11 else kwargs["output"])
    if _INITIAL_CHECKPOINT is None:
        raise RuntimeError("initial checkpoint was not configured")
    payload = torch.load(_INITIAL_CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"{_INITIAL_CHECKPOINT} must contain model_state")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = "student.candidate_personalization."
    invalid = [key for key in missing if not key.startswith(allowed)]
    if invalid or unexpected:
        raise RuntimeError(
            "checkpoint is incompatible with the acceleration model; "
            f"missing={invalid}, unexpected={list(unexpected)}"
        )
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": base.clone_state(model),
        "source_checkpoint": str(_INITIAL_CHECKPOINT),
    }, output / "initialized.pt")
    print(
        f"initialized from {_INITIAL_CHECKPOINT}; created {len(missing)} "
        "zero-safe candidate-adapter tensors"
    )
    return [{"phase": "candidate_initialization"}]


def staged_train_student_phase(
    phase: str,
    model: PersonalizedCompleteReachModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: dict[str, Any],
    epochs: int,
    context_samples: int,
    patch_length: int,
    lead_window: tuple[int, int],
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    mean_target: torch.Tensor,
    device: torch.device,
    output: Path,
    difficulty: Any,
    adaptive: bool,
    unfreeze_decoder: bool,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], float]:
    if phase not in {"student", "finetune"}:
        return _ORIGINAL_TRAIN_STUDENT_PHASE(
            phase, model, train_loader, validation_loader, config, epochs,
            context_samples, patch_length, lead_window, evaluation_leads,
            canvas_tensor, mean_target, device, output, difficulty, adaptive,
            unfreeze_decoder,
        )
    settings = config["model"]["candidate_personalization"]
    adapter_only = phase == "student"

    def selected_parameters(module: torch.nn.Module, enabled: bool) -> None:
        if module is model.student and enabled:
            _ORIGINAL_SET_TRAINABLE(module, False)
            _ORIGINAL_SET_TRAINABLE(model.student.candidate_personalization, True)
            if not adapter_only:
                for selected in (
                    model.student.endpoint_decoder,
                    model.student.teacher_latent_bridge,
                    model.student.emg_teacher_latent_bridge,
                    model.student.soft_routed_reach_heads,
                    model.student.emg_temporal_residual_head,
                    model.student.emg_acceleration_dynamics_head,
                ):
                    _ORIGINAL_SET_TRAINABLE(selected, True)
        elif module in {model.guidance, model.decoder} and enabled:
            _ORIGINAL_SET_TRAINABLE(module, False)
        else:
            _ORIGINAL_SET_TRAINABLE(module, enabled)

    original_lr = float(config["training"]["learning_rate"])
    if not adapter_only:
        config["training"]["learning_rate"] = original_lr * float(
            settings.get("head_learning_rate_factor", 0.10)
        )
    model.personalization_warmup = adapter_only
    base.set_trainable = selected_parameters
    try:
        return _ORIGINAL_TRAIN_STUDENT_PHASE(
            "candidate_adapter" if adapter_only else "candidate_head_finetune",
            model, train_loader, validation_loader, config, epochs,
            context_samples, patch_length, lead_window, evaluation_leads,
            canvas_tensor, mean_target, device, output, difficulty, adaptive,
            False,
        )
    finally:
        base.set_trainable = _ORIGINAL_SET_TRAINABLE
        model.personalization_warmup = False
        config["training"]["learning_rate"] = original_lr


def _has_option(name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in sys.argv[1:])


def _option_value(name: str, default: str) -> str:
    for index, value in enumerate(sys.argv[1:]):
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
        if value == name and index + 2 <= len(sys.argv[1:]):
            return sys.argv[1:][index + 1]
    return default


def _pop_option(name: str, default: str) -> str:
    arguments = sys.argv[1:]
    for index, value in enumerate(arguments):
        if value.startswith(name + "="):
            del sys.argv[index + 1]
            return value.split("=", 1)[1]
        if value == name and index + 1 < len(arguments):
            result = arguments[index + 1]
            del sys.argv[index + 1:index + 3]
            return result
    return default


def main() -> None:
    global _INITIAL_CHECKPOINT
    _INITIAL_CHECKPOINT = Path(_pop_option(
        "--initial-checkpoint", "runs/emg_acceleration_complete_reach/final.pt"
    ))
    help_requested = any(value in {"-h", "--help"} for value in sys.argv[1:])
    if not help_requested and not _INITIAL_CHECKPOINT.is_file():
        raise SystemExit(f"initial checkpoint not found: {_INITIAL_CHECKPOINT}")
    if not _has_option("--config"):
        sys.argv[1:1] = [
            "--config", "configs/tracked_personalized_complete_reach.yaml"
        ]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = ["--output-dir", "runs/personalized_complete_reach"]

    base.build_experiment_loaders = build_candidate_loaders
    base.make_distillation_window = complete.make_complete_reach_window
    base.milliseconds_to_samples = complete.milliseconds_to_samples
    base.evaluate = acceleration.residual.soft.evaluate
    base.train_teacher = load_initial_checkpoint
    base.train_student_phase = staged_train_student_phase
    channel.ChannelHorizonLatentDistillationModel = PersonalizedCompleteReachModel
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    print(
        "candidate personalization: training-only normalization -> frozen "
        "population encoder -> low-rank adapter -> low-rate output-head tuning"
    )
    channel.main()
    if _CANDIDATE_CALIBRATION is None:
        raise RuntimeError("candidate calibration was not constructed")
    output = Path(_option_value("--output-dir", "runs/personalized_complete_reach"))
    np.savez_compressed(
        output / "live_calibration.npz",
        emg_scale=_CANDIDATE_CALIBRATION[0],
        imu_center=_CANDIDATE_CALIBRATION[1],
        imu_scale=_CANDIDATE_CALIBRATION[2],
    )
    print(f"wrote train-only live calibration to {output / 'live_calibration.npz'}")


if __name__ == "__main__":
    main()
