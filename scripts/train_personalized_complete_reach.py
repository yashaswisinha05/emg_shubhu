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
_CANDIDATE_CALIBRATIONS: dict[
    str, tuple[np.ndarray, np.ndarray, np.ndarray]
] = {}


def _candidate_for_path(path: Path, prefixes: list[str]) -> str | None:
    for part in path.parts:
        lowered = part.lower()
        for prefix in prefixes:
            if (
                lowered == prefix
                or lowered.startswith(prefix + "_")
                or lowered.startswith(prefix + "-")
            ):
                return prefix
    return None


class CandidateDataset(TrackedTrajectoryDataset):
    """Apply the correct training-only calibration to each candidate."""

    def __init__(
        self,
        trials: list[Path],
        data_config: dict[str, Any],
        cache_dir: Path | None,
        trial_candidates: dict[str, str],
        candidate_index: dict[str, int],
        emg_scales: dict[str, np.ndarray],
        imu_statistics: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        super().__init__(
            trials, data_config, cache_dir, session_index={}, apply_emg_pca=False
        )
        self.trial_candidates = trial_candidates
        self.candidate_index = candidate_index
        self.candidate_emg_scales = emg_scales
        self.candidate_imu_statistics = imu_statistics

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = super().__getitem__(index)
        if result.get("unusable"):
            return result
        candidate = self.trial_candidates[str(self.trials[index])]
        scale = self.candidate_emg_scales.get(candidate)
        if scale is not None:
            result["emg"] = result["emg"] / torch.from_numpy(scale)
        result["emg"] = torch.from_numpy(
            apply_sensor_local_pca(result["emg"].numpy(), self.data_config)
        )
        statistics = self.candidate_imu_statistics.get(candidate)
        if statistics is not None:
            centre, spread = statistics
            result["imu"] = (
                result["imu"] - torch.from_numpy(centre)
            ) / torch.from_numpy(spread)
        result["session"] = self.candidate_index[candidate]
        return result


def build_candidate_loaders(
    config: dict[str, Any], root: str | Path, cache_dir: str | Path | None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    global _CANDIDATE_CALIBRATIONS
    prefixes = [
        str(value).strip().lower()
        for value in config["data"].get("include_session_prefixes", [])
        if str(value).strip()
    ]
    if not prefixes:
        raise ValueError("provide candidate folder names using --session-prefixes")
    grouped: dict[str, list[Path]] = {prefix: [] for prefix in prefixes}
    for paths in discover_trials(root).values():
        for path in paths:
            candidate = _candidate_for_path(path, prefixes)
            if candidate is not None:
                grouped[candidate].append(path)
    missing = [prefix for prefix, trials in grouped.items() if not trials]
    if missing:
        raise ValueError(
            "candidate prefix matched no trial_*.csv: " + ", ".join(missing)
        )

    validation_fraction = float(config["data"].get("validation_fraction", 0.2))
    test_fraction = float(config["data"].get("test_fraction", 0.2))
    train: list[Path] = []
    validation: list[Path] = []
    test: list[Path] = []
    training_by_candidate: dict[str, list[Path]] = {}
    seed = int(config.get("seed", 42))
    for index, prefix in enumerate(prefixes):
        selected = list(grouped[prefix])
        np.random.default_rng(seed + index).shuffle(selected)
        n_test = max(1, int(round(len(selected) * test_fraction)))
        n_validation = max(1, int(round(len(selected) * validation_fraction)))
        candidate_train = selected[n_test + n_validation:]
        if len(candidate_train) < 20:
            raise ValueError(
                f"only {len(candidate_train)} training trials remain for {prefix}"
            )
        training_by_candidate[prefix] = candidate_train
        train.extend(candidate_train)
        validation.extend(selected[n_test:n_test + n_validation])
        test.extend(selected[:n_test])

    # Strict evaluation: neither validation nor test samples contribute to
    # normalization or PCA fitting.
    scales: dict[str, np.ndarray] = {}
    statistics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    normalise_emg = bool(config["data"].get("emg_session_normalise", True))
    normalise_imu = bool(config["data"].get("imu_session_normalise", True))
    for prefix, candidate_train in training_by_candidate.items():
        if normalise_emg:
            found_scale = session_emg_scale(candidate_train, config["data"])
            if found_scale is not None:
                scales[prefix] = found_scale
        if normalise_imu:
            found_statistics = session_imu_statistics(
                candidate_train, config["data"]
            )
            if found_statistics is not None:
                statistics[prefix] = found_statistics
    trial_candidates = {
        str(path): prefix for prefix, trials in grouped.items() for path in trials
    }
    fit_training_emg_pca(
        train, config["data"], scales, trial_sessions=trial_candidates
    )
    emg_dim = raw_emg_feature_count(config["data"])
    imu_dim = imu_feature_count(config["data"])
    _CANDIDATE_CALIBRATIONS = {}
    for prefix in prefixes:
        emg_scale = scales.get(prefix, np.ones(emg_dim, dtype=np.float32))
        imu_centre, imu_scale = statistics.get(prefix, (
            np.zeros(imu_dim, dtype=np.float32),
            np.ones(imu_dim, dtype=np.float32),
        ))
        _CANDIDATE_CALIBRATIONS[prefix] = (
            np.asarray(emg_scale, dtype=np.float32),
            np.asarray(imu_centre, dtype=np.float32),
            np.asarray(imu_scale, dtype=np.float32),
        )
    candidate_settings = config["model"].get("candidate_personalization", {})
    config.setdefault("virtual_leader", {})["session_count"] = int(
        candidate_settings.get("source_session_count", len(prefixes))
    )
    cache = Path(cache_dir) if cache_dir else None
    batch_size = int(config["training"].get("batch_size", 16))
    workers = int(config["training"].get("num_workers", 0))

    def loader(selected: list[Path], shuffle: bool) -> DataLoader:
        return DataLoader(
            CandidateDataset(
                selected,
                config["data"],
                cache,
                trial_candidates,
                {prefix: index for index, prefix in enumerate(prefixes)},
                scales,
                statistics,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=collate_tracked,
            drop_last=False,
        )

    for prefix in prefixes:
        total = len(grouped[prefix])
        training = len(training_by_candidate[prefix])
        print(
            f"candidate {prefix}: {total} total | {training} train | "
            f"{total - training} validation+untouched-test"
        )
    print(
        f"combined: {len(train)} train | {len(validation)} validation | "
        f"{len(test)} untouched test"
    )
    print("normalization fitted separately on each candidate's training trials")
    return loader(train, True), loader(validation, False), loader(test, False)


def save_candidate_calibration(output: str | Path) -> list[Path]:
    """Write exact train-only statistics for every selected candidate."""
    if not _CANDIDATE_CALIBRATIONS:
        raise RuntimeError("candidate calibration was not constructed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for prefix, calibration in _CANDIDATE_CALIBRATIONS.items():
        safe = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in prefix
        )
        name = (
            "live_calibration.npz"
            if len(_CANDIDATE_CALIBRATIONS) == 1
            else f"live_calibration_{safe}.npz"
        )
        path = output / name
        np.savez_compressed(
            path,
            emg_scale=calibration[0],
            imu_center=calibration[1],
            imu_scale=calibration[2],
        )
        written.append(path)
        print(f"wrote {prefix} train-only live calibration to {path}")
    return written


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
    output = Path(_option_value("--output-dir", "runs/personalized_complete_reach"))
    save_candidate_calibration(output)


if __name__ == "__main__":
    main()
