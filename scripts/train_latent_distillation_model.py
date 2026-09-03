#!/usr/bin/env python3
"""Train a privileged VIVE teacher, then distil it into EMG+IMU.

This is a new experimental path.  It does not import or modify the existing
VAE-discriminator model.  The implementation adapts the useful mechanism in
MUSIC (dense, multi-step latent distillation) to wearable intent inference:

  Stage 1: a training-only teacher compresses the true future VIVE trajectory
           and learns a shared trajectory + screen-destination decoder.
  Stage 2: an EMG+IMU student learns the teacher distribution and behavior
           while the teacher and decoder are frozen.
  Stage 3: optional low-rate decoder fine-tuning improves the deployable
           student without allowing VIVE into its encoder.

The test prediction is always produced by
``student_forward(emg, imu, time_mask)``.  VIVE appears only as privileged
training supervision and offline evaluation ground truth.

Example:

    python scripts/train_latent_distillation_model.py \\
      --root "/media/.../emg_imu_vive" \\
      --config configs/tracked_latent_distillation.yaml \\
      --cache-dir artifacts/tracked_cache_posture \\
      --device cuda --teacher-epochs 10 --epochs 20 --finetune-epochs 5 \\
      --lead-window-ms 50 400 \\
      --output-dir runs/latent_distillation_emg_imu
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    CANVAS_COLUMNS,
    TrackedTrajectoryDataset,
    collate_tracked,
    discover_trials,
    emg_feature_count,
    imu_feature_count,
    session_emg_scale,
    session_imu_statistics,
    split_sessions,
)
from emg_touch.grid_training import grid_point_loss  # noqa: E402
from emg_touch.models.latent_distillation import (  # noqa: E402
    WearableLatentDistillationModel,
    diagonal_gaussian_kl,
    standard_normal_kl,
)
from emg_touch.utils import choose_device, save_json, seed_everything  # noqa: E402


def canvas_from_disk(root: str) -> tuple[float, float] | None:
    for path in sorted(Path(root).rglob("trial_*.csv"))[:1]:
        try:
            frame = pd.read_csv(path, nrows=64)
        except Exception:  # noqa: BLE001
            return None
        if not all(name in frame.columns for name in CANVAS_COLUMNS):
            return None
        values = (
            frame[list(CANVAS_COLUMNS)]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .to_numpy()
        )
        if len(values):
            return float(values[-1][0]), float(values[-1][1])
    return None


def effective_rate(config: dict[str, Any]) -> float:
    return float(config["data"]["sample_rate_hz"]) / max(
        1, int(config["data"].get("decimation", 10))
    )


def milliseconds_to_samples(milliseconds: float, rate: float) -> int:
    return max(1, int(round(float(milliseconds) * rate / 1000.0)))


def select_sessions(
    sessions: dict[str, list[Path]], prefixes: list[str] | tuple[str, ...]
) -> dict[str, list[Path]]:
    """Restrict trials by a matching session key or ancestor folder.

    Some exported datasets put ``trial_*.csv`` one or more directories below
    the ``dev_a1_vive__...`` folder. The generic loader groups trials by their
    immediate parent, so checking only its dictionary key misses those
    exports. Regrouping by the matched ancestor also keeps normalization
    separate for each recording session.
    """
    cleaned = tuple(
        str(prefix).strip().lower()
        for prefix in prefixes
        if str(prefix).strip()
    )
    if not cleaned:
        return sessions

    def matches(label: str, prefix: str) -> bool:
        lowered = label.lower()
        return (
            lowered == prefix
            or lowered.startswith(prefix + "_")
            or lowered.startswith(prefix + "-")
        )

    selected: dict[str, list[Path]] = {}
    found: set[str] = set()
    for name, trials in sessions.items():
        for path in trials:
            owner: str | None = None
            for prefix in cleaned:
                if matches(name, prefix):
                    owner = name
                    found.add(prefix)
                    break
            if owner is None:
                for part in reversed(path.parts):
                    matching = [prefix for prefix in cleaned if matches(part, prefix)]
                    if matching:
                        owner = part
                        found.update(matching)
                        break
            if owner is not None:
                selected.setdefault(owner, []).append(path)

    missing = [prefix for prefix in cleaned if prefix not in found]
    if missing:
        raise ValueError(
            "session prefix(es) matched no dataset folder or trial ancestor: "
            + ", ".join(missing)
        )
    if len(selected) < 3:
        raise ValueError(
            f"session selection retained only {len(selected)} session(s); "
            "at least three are required for train/validation/test splits"
        )
    return selected


class ExperimentTrackedTrajectoryDataset(TrackedTrajectoryDataset):
    """Tracked dataset normalized by the selected ancestor session."""

    def __init__(
        self,
        trials: list[Path],
        data_config: dict[str, Any],
        cache_dir: Path | None,
        trial_sessions: dict[str, str],
        session_index: dict[str, int],
        emg_scales: dict[str, np.ndarray],
        imu_statistics: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        # Leave the base normalization maps empty; __getitem__ applies the
        # correct ancestor-session maps after loading cached raw features.
        super().__init__(trials, data_config, cache_dir, session_index={})
        self.trial_sessions = trial_sessions
        self.experiment_session_index = session_index
        self.experiment_emg_scales = emg_scales
        self.experiment_imu_statistics = imu_statistics

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = super().__getitem__(index)
        if result.get("unusable"):
            return result
        path = self.trials[index]
        session = self.trial_sessions[str(path)]
        scale = self.experiment_emg_scales.get(session)
        if scale is not None:
            result["emg"] = result["emg"] / torch.from_numpy(scale)
        statistics = self.experiment_imu_statistics.get(session)
        if statistics is not None:
            centre, spread = statistics
            result["imu"] = (
                result["imu"] - torch.from_numpy(centre)
            ) / torch.from_numpy(spread)
        result["session"] = int(self.experiment_session_index[session])
        return result


def build_experiment_loaders(
    config: dict[str, Any], root: str | Path, cache_dir: str | Path | None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build loaders after filtering, including normalization statistics.

    This local loader keeps the selection isolated to the new experiment.
    Excluded sessions are not used for splitting, EMG scaling, IMU
    normalization, or sampling.
    """
    sessions = discover_trials(root)
    if not sessions:
        raise ValueError(f"no trial_*.csv found under {root}")
    prefixes = list(config["data"].get("include_session_prefixes", []))
    sessions = select_sessions(sessions, prefixes)
    train, validation, test = split_sessions(sessions, config)
    session_index = {name: index for index, name in enumerate(sorted(sessions))}
    trial_sessions = {
        str(path): name for name, trials in sessions.items() for path in trials
    }

    scales: dict[str, np.ndarray] = {}
    if bool(config["data"].get("emg_session_normalise", True)):
        for name, trials in sessions.items():
            scale = session_emg_scale(trials, config["data"])
            if scale is not None:
                scales[name] = scale

    imu_statistics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if bool(config["data"].get("imu_session_normalise", True)):
        for name, trials in sessions.items():
            found = session_imu_statistics(trials, config["data"])
            if found is not None:
                imu_statistics[name] = found

    config.setdefault("virtual_leader", {})["session_count"] = len(session_index)
    batch_size = int(config["training"].get("batch_size", 16))
    workers = int(config["training"].get("num_workers", 0))

    def loader(trials: list[Path], shuffle: bool) -> DataLoader:
        dataset = ExperimentTrackedTrajectoryDataset(
            trials,
            config["data"],
            Path(cache_dir) if cache_dir else None,
            trial_sessions,
            session_index,
            scales,
            imu_statistics,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=collate_tracked,
            drop_last=False,
        )

    print("selected dataset sessions:")
    for name, trials in sorted(sessions.items()):
        print(f"  {name}: {len(trials)} trials")
    return loader(train, True), loader(validation, False), loader(test, False)


def make_distillation_window(
    batch: dict[str, Any],
    context_samples: int,
    patch_length: int,
    teacher_steps: int,
    generator: np.random.Generator,
    trajectory_limit_m: float,
    velocity_scale_mps: float,
    fallback_canvas: torch.Tensor | None = None,
    lead_window: tuple[int, int] | None = None,
    fixed_lead: int | None = None,
    cutoffs_per_trial: int = 1,
) -> dict[str, Any] | None:
    """Create causal student inputs and a separate privileged teacher target.

    Student history ends strictly before ``cut``.  Teacher features contain
    VIVE positions/velocities from ``cut`` through touch and are returned under
    a separate key that is accepted only by ``teacher_forward``.
    """
    lengths = batch["lengths"]
    chosen: list[tuple[int, int, int, int]] = []
    for row in range(len(lengths)):
        touch = int(lengths[row]) - 1
        for _ in range(cutoffs_per_trial):
            if fixed_lead is not None:
                lead = int(fixed_lead)
            elif lead_window is not None:
                low, high = lead_window
                lead = int(generator.integers(low, high + 1))
            else:
                # Default to a random prediction instant after one sample and
                # before touch. Explicit lead windows are recommended.
                lead = int(generator.integers(1, max(2, touch)))
            cut = touch - lead
            if cut <= 0 or cut > touch:
                continue
            chosen.append((row, cut, touch, lead))
    if not chosen:
        return None

    prefix = max(int(context_samples), int(patch_length))
    device = batch["emg"].device
    emg_windows, imu_windows, masks = [], [], []
    teacher_features, trajectory_targets = [], []
    source_paths: list[str] = []

    for row, cut, touch, _ in chosen:
        start = max(0, cut - prefix)
        mask = torch.zeros(prefix, dtype=torch.bool, device=device)
        available_count = cut - start
        mask[-available_count:] = True
        masks.append(mask)

        for source, destination in (
            (batch["emg"], emg_windows), (batch["imu"], imu_windows)
        ):
            available = source[row, start:cut]
            padded = source.new_zeros(prefix, source.size(-1))
            padded[-available.size(0):] = available
            destination.append(padded)

        indices = torch.linspace(
            cut, touch, teacher_steps, device=device
        ).round().long().clamp(cut, touch)
        origin = batch["position"][row, cut - 1]
        relative_position = batch["position"][row].index_select(0, indices) - origin
        future_velocity = batch["velocity"][row].index_select(0, indices)
        trajectory_targets.append(relative_position)
        teacher_features.append(torch.cat([
            relative_position / max(float(trajectory_limit_m), 1e-6),
            future_velocity / max(float(velocity_scale_mps), 1e-6),
        ], dim=-1))
        paths = batch.get("paths")
        source_paths.append(str(paths[row]) if paths is not None else str(row))

    rows = torch.tensor([item[0] for item in chosen], dtype=torch.long, device=device)
    window: dict[str, Any] = {
        "emg": torch.stack(emg_windows),
        "imu": torch.stack(imu_windows),
        "time_mask": torch.stack(masks),
        "teacher_features": torch.stack(teacher_features),
        "trajectory_target": torch.stack(trajectory_targets),
        "target": batch["screen_target"].index_select(0, rows),
        "loss_weight": torch.ones(len(chosen), dtype=torch.float32, device=device),
        "lead_samples": torch.tensor(
            [item[3] for item in chosen], dtype=torch.long, device=device
        ),
        "samples_past_onset": torch.tensor(
            [item[1] - int(batch["onset"][item[0]]) for item in chosen],
            dtype=torch.long,
            device=device,
        ),
        "source_paths": source_paths,
    }
    if "canvas" in batch:
        window["canvas_size"] = batch["canvas"].index_select(0, rows)
    elif fallback_canvas is not None:
        window["canvas_size"] = fallback_canvas.to(device).unsqueeze(0).expand(
            len(chosen), -1
        )
    else:
        raise ValueError("canvas size is required for the screen-point loss")
    return window


def trajectory_errors(
    prediction: torch.Tensor, target: torch.Tensor, epsilon_m: float
) -> torch.Tensor:
    return torch.sqrt(
        (prediction - target).square().sum(dim=-1) + float(epsilon_m) ** 2
    ).mean(dim=-1)


def teacher_objective(
    outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
    kl_weight: float,
) -> dict[str, torch.Tensor]:
    settings = config["distillation"]
    point = grid_point_loss(outputs, window, config)
    per_sample_trajectory = trajectory_errors(
        outputs["trajectory"], window["trajectory_target"],
        float(settings.get("trajectory_epsilon_m", 0.002)),
    )
    trajectory = per_sample_trajectory.mean()
    kl = standard_normal_kl(outputs["mu"], outputs["log_variance"])
    total = (
        point["loss"]
        + float(settings.get("teacher_trajectory_weight", 2.0)) * trajectory
        + float(kl_weight) * kl
    )
    return {
        "loss": total,
        "point": point["loss"].detach(),
        "trajectory": trajectory.detach(),
        "kl": kl.detach(),
        "per_sample_trajectory": per_sample_trajectory.detach(),
    }


def student_objective(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    settings = config["distillation"]
    epsilon = float(settings.get("trajectory_epsilon_m", 0.002))
    point = grid_point_loss(outputs, window, config)
    trajectory = trajectory_errors(
        outputs["trajectory"], window["trajectory_target"], epsilon
    ).mean()
    latent = diagonal_gaussian_kl(
        outputs["mu"], outputs["log_variance"],
        teacher_outputs["mu"], teacher_outputs["log_variance"],
        float(settings.get("teacher_sigma_floor", 0.05)),
    )
    prediction_distillation = F.smooth_l1_loss(
        outputs["prediction"], teacher_outputs["prediction"].detach()
    )
    trajectory_distillation = trajectory_errors(
        outputs["trajectory"], teacher_outputs["trajectory"].detach(), epsilon
    ).mean()

    emg_outputs = outputs["emg_only"]
    emg_point = grid_point_loss(emg_outputs, window, config)
    emg_trajectory = trajectory_errors(
        emg_outputs["trajectory"], window["trajectory_target"], epsilon
    ).mean()
    emg_latent = diagonal_gaussian_kl(
        emg_outputs["mu"], emg_outputs["log_variance"],
        teacher_outputs["mu"], teacher_outputs["log_variance"],
        float(settings.get("teacher_sigma_floor", 0.05)),
    )
    imu_tracking = trajectory_errors(
        outputs["imu_trajectory"], window["trajectory_target"], epsilon
    ).mean()

    total = (
        point["loss"]
        + float(settings.get("student_trajectory_weight", 2.0)) * trajectory
        + float(settings.get("latent_distillation_weight", 1.0)) * latent
        + float(settings.get("prediction_distillation_weight", 0.25))
        * prediction_distillation
        + float(settings.get("trajectory_distillation_weight", 1.0))
        * trajectory_distillation
        + float(settings.get("emg_only_weight", 0.5))
        * (
            emg_point["loss"]
            + float(settings.get("student_trajectory_weight", 2.0)) * emg_trajectory
        )
        + float(settings.get("emg_latent_weight", 0.5)) * emg_latent
        + float(settings.get("imu_tracking_weight", 0.25)) * imu_tracking
    )
    return {
        "loss": total,
        "point": point["loss"].detach(),
        "trajectory": trajectory.detach(),
        "latent": latent.detach(),
        "prediction_distillation": prediction_distillation.detach(),
        "trajectory_distillation": trajectory_distillation.detach(),
        "emg_point": emg_point["loss"].detach(),
        "emg_trajectory": emg_trajectory.detach(),
        "emg_latent": emg_latent.detach(),
        "imu_tracking": imu_tracking.detach(),
    }


class AdaptiveTrialDifficulty:
    """EMA difficulty memory used to oversample hard trials with a safe cap."""

    def __init__(self, alpha: float, uniform_mix: float, power: float, max_ratio: float):
        self.alpha = float(alpha)
        self.uniform_mix = float(uniform_mix)
        self.power = float(power)
        self.max_ratio = float(max_ratio)
        if self.max_ratio <= 0.0:
            raise ValueError("adaptive sampling max_ratio must be positive")
        self.scores: dict[str, float] = {}

    def update(self, paths: list[str], errors: torch.Tensor) -> None:
        for path, error in zip(paths, errors.detach().cpu().tolist()):
            value = max(float(error), 1e-6)
            previous = self.scores.get(path, value)
            self.scores[path] = (1.0 - self.alpha) * previous + self.alpha * value

    def weights_for(self, trials: list[Path]) -> torch.Tensor:
        if not self.scores:
            return torch.ones(len(trials), dtype=torch.double)
        fallback = float(np.median(list(self.scores.values())))
        values = np.asarray(
            [self.scores.get(str(path), fallback) for path in trials], dtype=np.float64
        )
        median = max(float(np.median(values)), 1e-6)
        hard = np.power(np.maximum(values / median, 1e-6), self.power)
        hard = np.clip(hard, 1.0 / self.max_ratio, self.max_ratio)
        mixed = (1.0 - self.uniform_mix) + self.uniform_mix * hard
        return torch.as_tensor(mixed, dtype=torch.double)


def adaptive_train_loader(
    base_loader: DataLoader,
    difficulty: AdaptiveTrialDifficulty,
    seed: int,
) -> DataLoader:
    dataset = base_loader.dataset
    trials = list(getattr(dataset, "trials"))
    generator = torch.Generator().manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        difficulty.weights_for(trials),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=base_loader.batch_size,
        sampler=sampler,
        num_workers=base_loader.num_workers,
        collate_fn=base_loader.collate_fn,
        drop_last=False,
        pin_memory=base_loader.pin_memory,
    )


def _append(store: dict[str, list[float]], key: str, values: torch.Tensor) -> None:
    store.setdefault(key, []).extend(values.detach().cpu().tolist())


@torch.no_grad()
def evaluate(
    model: WearableLatentDistillationModel,
    loader: DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    mean_target: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    settings = config["distillation"]
    teacher_steps = int(config["model"]["teacher_trajectory_steps"])
    trajectory_limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    epsilon = float(settings.get("trajectory_epsilon_m", 0.002))
    totals: dict[str, list[float]] = {}
    generator = np.random.default_rng(0)

    for batch in loader:
        if batch is None:
            continue
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        for lead in evaluation_leads:
            window = make_distillation_window(
                batch, context_samples, patch_length, teacher_steps, generator,
                trajectory_limit, velocity_scale, canvas_tensor,
                fixed_lead=lead,
            )
            if window is None:
                continue
            teacher = model.teacher_forward(window["teacher_features"], sample=False)
            student = model.student_forward(
                window["emg"], window["imu"], window["time_mask"],
                sample=False, include_emg_only=True,
            )
            interventions = {
                "student": student,
                "emg_only": student["emg_only"],
                "without_emg": model.student_forward(
                    torch.zeros_like(window["emg"]), window["imu"],
                    window["time_mask"], sample=False,
                ),
                "without_imu": model.student_forward(
                    window["emg"], torch.zeros_like(window["imu"]),
                    window["time_mask"], sample=False,
                ),
            }
            if window["emg"].size(0) > 1:
                interventions["shuffled_emg"] = model.student_forward(
                    torch.roll(window["emg"], shifts=1, dims=0), window["imu"],
                    window["time_mask"], sample=False,
                )

            for name, output in {"teacher": teacher, **interventions}.items():
                pixel = (
                    (output["prediction"] - window["target"])
                    * window["canvas_size"]
                ).norm(dim=-1)
                trajectory = 100.0 * trajectory_errors(
                    output["trajectory"], window["trajectory_target"], epsilon
                )
                _append(totals, f"{name}_px", pixel)
                _append(totals, f"{name}_trajectory_cm", trajectory)
                _append(totals, f"{name}_lead_{lead}_px", pixel)

            mean = mean_target.to(device).unsqueeze(0).expand_as(window["target"])
            _append(
                totals,
                "mean_px",
                ((mean - window["target"]) * window["canvas_size"]).norm(dim=-1),
            )
            latent_rmse = torch.sqrt(
                (student["mu"] - teacher["mu"]).square().mean(dim=-1)
            )
            _append(totals, "latent_rmse", latent_rmse)
            _append(
                totals,
                "imu_tracking_trajectory_cm",
                100.0 * trajectory_errors(
                    student["imu_trajectory"], window["trajectory_target"], epsilon
                ),
            )

    return {name: float(np.mean(values)) for name, values in totals.items()}


@torch.no_grad()
def evaluate_teacher(
    model: WearableLatentDistillationModel,
    loader: DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    lead: int,
    canvas_tensor: torch.Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    point, trajectory = [], []
    generator = np.random.default_rng(0)
    teacher_steps = int(config["model"]["teacher_trajectory_steps"])
    trajectory_limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    epsilon = float(config["distillation"].get("trajectory_epsilon_m", 0.002))
    for batch in loader:
        if batch is None:
            continue
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        window = make_distillation_window(
            batch, context_samples, patch_length, teacher_steps, generator,
            trajectory_limit, velocity_scale, canvas_tensor, fixed_lead=lead,
        )
        if window is None:
            continue
        output = model.teacher_forward(window["teacher_features"], sample=False)
        point.extend((
            (output["prediction"] - window["target"]) * window["canvas_size"]
        ).norm(dim=-1).cpu().tolist())
        trajectory.extend((100.0 * trajectory_errors(
            output["trajectory"], window["trajectory_target"], epsilon
        )).cpu().tolist())
    return {
        "teacher_px": float(np.mean(point)) if point else float("inf"),
        "teacher_trajectory_cm": float(np.mean(trajectory)) if trajectory else float("inf"),
    }


def training_mean_target(loader: DataLoader) -> torch.Tensor:
    values = []
    for batch in loader:
        if batch is not None and "screen_target" in batch:
            values.extend(batch["screen_target"].unbind(0))
    return torch.stack(values).mean(dim=0).cpu() if values else torch.tensor([0.5, 0.5])


def set_trainable(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def train_teacher(
    model: WearableLatentDistillationModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: dict[str, Any],
    epochs: int,
    context_samples: int,
    patch_length: int,
    lead_window: tuple[int, int],
    selection_lead: int,
    canvas_tensor: torch.Tensor | None,
    device: torch.device,
    output: Path,
    difficulty: AdaptiveTrialDifficulty,
    adaptive: bool,
) -> list[dict[str, float]]:
    set_trainable(model.teacher, True)
    set_trainable(model.decoder, True)
    set_trainable(model.student, False)
    optimizer = torch.optim.AdamW(
        list(model.teacher.parameters()) + list(model.decoder.parameters()),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    settings = config["distillation"]
    warmup = max(1, int(settings.get("teacher_kl_warmup_epochs", 5)))
    maximum_kl = float(settings.get("teacher_kl_weight", 1e-4))
    teacher_steps = int(config["model"]["teacher_trajectory_steps"])
    trajectory_limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    clip = float(config["training"].get("gradient_clip_norm", 1.0))
    cutoffs = int(config["training"].get("cutoffs_per_trial", 3))
    history, best_state = [], None
    best = float("inf")
    generator = np.random.default_rng(int(config.get("seed", 42)))

    for epoch in range(1, epochs + 1):
        model.train()
        model.student.eval()
        loader = adaptive_train_loader(
            train_loader, difficulty, int(config.get("seed", 42)) + epoch
        ) if adaptive else train_loader
        progress = min(1.0, epoch / warmup)
        kl_weight = maximum_kl * progress
        running: dict[str, list[float]] = {
            "loss": [], "point": [], "trajectory": [], "kl": []
        }
        for batch in tqdm(loader, desc=f"teacher {epoch}"):
            if batch is None:
                continue
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            window = make_distillation_window(
                batch, context_samples, patch_length, teacher_steps, generator,
                trajectory_limit, velocity_scale, canvas_tensor,
                lead_window=lead_window, cutoffs_per_trial=cutoffs,
            )
            if window is None:
                continue
            outputs = model.teacher_forward(
                window["teacher_features"], sample=True, noise_scale=progress
            )
            losses = teacher_objective(outputs, window, config, kl_weight)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.teacher.parameters()) + list(model.decoder.parameters()), clip
            )
            optimizer.step()
            pixel = (
                (outputs["prediction"].detach() - window["target"])
                * window["canvas_size"]
            ).norm(dim=-1)
            difficulty.update(window["source_paths"], pixel)
            for name in running:
                running[name].append(float(losses[name].detach()))

        validation = evaluate_teacher(
            model, validation_loader, config, context_samples, patch_length,
            selection_lead, canvas_tensor, device,
        )
        record = {
            "phase": "teacher", "epoch": epoch,
            **{f"train_{name}": float(np.mean(values or [0.0]))
               for name, values in running.items()},
            **validation,
        }
        history.append(record)
        print(
            f"teacher epoch={epoch} loss={record['train_loss']:.4f} "
            f"traj={record['train_trajectory'] * 100:.2f}cm "
            f"kl={record['train_kl']:.4f} | "
            f"val={validation['teacher_px']:.1f}px/"
            f"{validation['teacher_trajectory_cm']:.2f}cm"
        )
        if validation["teacher_px"] < best:
            best = validation["teacher_px"]
            best_state = {
                "teacher": clone_state(model.teacher),
                "decoder": clone_state(model.decoder),
            }
            torch.save(
                {**best_state, "config": config, "validation": validation},
                output / "teacher_best.pt",
            )
    if best_state is None:
        raise RuntimeError("teacher training produced no valid validation windows")
    model.teacher.load_state_dict(best_state["teacher"])
    model.decoder.load_state_dict(best_state["decoder"])
    return history


def train_student_phase(
    phase: str,
    model: WearableLatentDistillationModel,
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
    difficulty: AdaptiveTrialDifficulty,
    adaptive: bool,
    unfreeze_decoder: bool,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], float]:
    set_trainable(model.teacher, False)
    set_trainable(model.student, True)
    set_trainable(model.decoder, unfreeze_decoder)
    learning_rate = float(config["training"]["learning_rate"])
    if unfreeze_decoder:
        parameters: Any = [
            {"params": model.student.parameters(), "lr": learning_rate},
            {
                "params": model.decoder.parameters(),
                "lr": learning_rate * float(
                    config["distillation"].get("decoder_finetune_lr_factor", 0.1)
                ),
            },
        ]
    else:
        parameters = model.student.parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"].get("scheduler_factor", 0.5)),
        patience=int(config["training"].get("scheduler_patience", 3)),
        min_lr=float(config["training"].get("minimum_learning_rate", 1e-6)),
    )
    teacher_steps = int(config["model"]["teacher_trajectory_steps"])
    trajectory_limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    noise_scale = float(config["distillation"].get("student_noise_scale", 0.5))
    cutoffs = int(config["training"].get("cutoffs_per_trial", 3))
    clip = float(config["training"].get("gradient_clip_norm", 1.0))
    patience = int(config["training"].get("early_stopping_patience", 7))
    generator = np.random.default_rng(
        int(config.get("seed", 42)) + (1000 if unfreeze_decoder else 100)
    )
    # Keep the incoming distilled checkpoint unless this phase actually
    # improves it. In particular, decoder fine-tuning must not replace a good
    # student merely because its local ``best`` was reset to infinity.
    baseline = evaluate(
        model, validation_loader, config, context_samples, patch_length,
        evaluation_leads, canvas_tensor, mean_target, device,
    )
    history, best_state = [], clone_state(model)
    best_validation = baseline
    best = baseline.get("student_px", float("inf"))
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        model.teacher.eval()
        if not unfreeze_decoder:
            model.decoder.eval()
        loader = adaptive_train_loader(
            train_loader, difficulty,
            int(config.get("seed", 42)) + 10000 + epoch,
        ) if adaptive else train_loader
        running: dict[str, list[float]] = {
            key: [] for key in (
                "loss", "point", "trajectory", "latent",
                "prediction_distillation", "trajectory_distillation",
                "emg_point", "emg_trajectory", "emg_latent", "imu_tracking",
            )
        }
        for batch in tqdm(loader, desc=f"{phase} {epoch}"):
            if batch is None:
                continue
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            window = make_distillation_window(
                batch, context_samples, patch_length, teacher_steps, generator,
                trajectory_limit, velocity_scale, canvas_tensor,
                lead_window=lead_window, cutoffs_per_trial=cutoffs,
            )
            if window is None:
                continue
            with torch.no_grad():
                teacher = model.teacher_forward(
                    window["teacher_features"], sample=False
                )
            student = model.student_forward(
                window["emg"], window["imu"], window["time_mask"],
                sample=True, noise_scale=noise_scale,
                include_emg_only=True, apply_imu_dropout=True,
            )
            losses = student_objective(student, teacher, window, config)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            trainable = [parameter for parameter in model.parameters()
                         if parameter.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, clip)
            optimizer.step()
            pixel = (
                (student["prediction"].detach() - window["target"])
                * window["canvas_size"]
            ).norm(dim=-1)
            difficulty.update(window["source_paths"], pixel)
            for name in running:
                running[name].append(float(losses[name].detach()))

        validation = evaluate(
            model, validation_loader, config, context_samples, patch_length,
            evaluation_leads, canvas_tensor, mean_target, device,
        )
        selection = validation.get("student_px", float("inf"))
        scheduler.step(selection)
        record = {
            "phase": phase, "epoch": epoch,
            **{f"train_{name}": float(np.mean(values or [0.0]))
               for name, values in running.items()},
            **validation,
        }
        history.append(record)
        print(
            f"{phase} epoch={epoch} loss={record['train_loss']:.4f} "
            f"latent={record['train_latent']:.4f} | "
            f"val student={selection:.1f}px "
            f"teacher={validation.get('teacher_px', float('nan')):.1f}px "
            f"traj={validation.get('student_trajectory_cm', float('nan')):.2f}cm "
            f"EMG gain={validation.get('without_emg_px', selection) - selection:+.1f}px"
        )
        if selection < best:
            best, stale = selection, 0
            best_state = clone_state(model)
            best_validation = validation
            torch.save(
                {"model_state": best_state, "config": config,
                 "validation": validation, "phase": phase},
                output / "best.pt",
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"{phase}: early stop after {stale} stale epochs")
                break
    model.load_state_dict(best_state)
    # Also covers a phase with zero improvement: the retained incoming state
    # remains a complete, explicitly recorded wearable checkpoint.
    torch.save(
        {
            "model_state": best_state,
            "config": config,
            "validation": best_validation,
            "phase": phase,
        },
        output / "best.pt",
    )
    return history, best_state, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--config", default="configs/tracked_latent_distillation.yaml"
    )
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache_posture")
    parser.add_argument("--output-dir", default="runs/latent_distillation_emg_imu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--teacher-epochs", type=int)
    parser.add_argument("--epochs", type=int, help="Student distillation epochs")
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument(
        "--lead-window-ms", type=float, nargs=2, metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--session-prefixes",
        nargs="+",
        help="Override data.include_session_prefixes (for example dev_a1 dev_a2)",
    )
    parser.add_argument("--no-adaptive-sampling", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.teacher_epochs is not None:
        config["distillation"]["teacher_epochs"] = args.teacher_epochs
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.finetune_epochs is not None:
        config["distillation"]["finetune_epochs"] = args.finetune_epochs
    if args.lead_window_ms is not None:
        config["distillation"]["lead_window_ms"] = list(args.lead_window_ms)
    if args.session_prefixes is not None:
        config["data"]["include_session_prefixes"] = args.session_prefixes

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = choose_device(args.device)
    train_loader, validation_loader, test_loader = build_experiment_loaders(
        config, args.root, Path(args.cache_dir)
    )
    rate = effective_rate(config)
    model_config = config["model"]
    context_samples = max(
        int(model_config["patch_length"]),
        milliseconds_to_samples(float(model_config.get("context_ms", 2000.0)), rate),
    )
    lead_ms = config["distillation"].get("lead_window_ms", [50.0, 400.0])
    low_ms, high_ms = sorted(map(float, lead_ms))
    lead_window = (
        milliseconds_to_samples(low_ms, rate),
        milliseconds_to_samples(high_ms, rate),
    )
    evaluation_leads = tuple(dict.fromkeys(
        milliseconds_to_samples(value, rate)
        for value in config["distillation"].get(
            "evaluation_leads_ms", [50.0, 100.0, 200.0, 300.0, 400.0]
        )
    ))
    selection_lead = milliseconds_to_samples(
        float(config["distillation"].get("selection_lead_ms", 200.0)), rate
    )

    fallback = canvas_from_disk(args.root)
    canvas_tensor = (
        torch.tensor(fallback, dtype=torch.float32, device=device)
        if fallback else None
    )
    mean_target = training_mean_target(train_loader)
    model = WearableLatentDistillationModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    ).to(device)

    student_parameters = set(
        inspect.signature(model.student_forward).parameters
    )
    forbidden = {"position", "velocity", "trajectory_features"}
    assert not (student_parameters & forbidden)
    print(
        "deployment input check: student_forward accepts EMG + IMU + mask only; "
        "VIVE is confined to the separate teacher_forward and labels"
    )
    print(
        f"train {len(train_loader.dataset)} | val {len(validation_loader.dataset)} "
        f"| test {len(test_loader.dataset)} trials"
    )
    print(
        f"context {context_samples} samples ({context_samples / rate:.2f}s), "
        f"lead {low_ms:.0f}-{high_ms:.0f}ms, "
        f"latent {model_config['latent_dim']}D, "
        f"teacher chunk {model_config['teacher_trajectory_steps']} steps"
    )
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adaptive_settings = config["distillation"].get("adaptive_sampling", {})
    adaptive = bool(adaptive_settings.get("enabled", True)) and not args.no_adaptive_sampling
    difficulty = AdaptiveTrialDifficulty(
        alpha=float(adaptive_settings.get("ema_alpha", 0.5)),
        uniform_mix=float(adaptive_settings.get("uniform_mix", 0.7)),
        power=float(adaptive_settings.get("power", 1.0)),
        max_ratio=float(adaptive_settings.get("max_ratio", 4.0)),
    )

    history = train_teacher(
        model, train_loader, validation_loader, config,
        int(config["distillation"].get("teacher_epochs", 10)),
        context_samples, int(model_config["patch_length"]), lead_window,
        selection_lead, canvas_tensor, device, output, difficulty, adaptive,
    )
    student_history, _, _ = train_student_phase(
        "student", model, train_loader, validation_loader, config,
        int(config["training"]["epochs"]), context_samples,
        int(model_config["patch_length"]), lead_window, evaluation_leads,
        canvas_tensor, mean_target, device, output, difficulty, adaptive,
        unfreeze_decoder=False,
    )
    history.extend(student_history)

    finetune_epochs = int(config["distillation"].get("finetune_epochs", 0))
    if finetune_epochs > 0:
        finetune_history, _, _ = train_student_phase(
            "finetune", model, train_loader, validation_loader, config,
            finetune_epochs, context_samples, int(model_config["patch_length"]),
            lead_window, evaluation_leads, canvas_tensor, mean_target, device,
            output, difficulty, adaptive, unfreeze_decoder=True,
        )
        history.extend(finetune_history)

    test = evaluate(
        model, test_loader, config, context_samples,
        int(model_config["patch_length"]), evaluation_leads,
        canvas_tensor, mean_target, device,
    )
    print("\n=== wearable-only student test ===")
    for name in (
        "student", "emg_only", "without_emg", "shuffled_emg", "without_imu",
        "mean", "teacher",
    ):
        key = f"{name}_px"
        if key in test:
            suffix = " (training-only oracle)" if name == "teacher" else ""
            print(f"  {name:13}: {test[key]:7.1f} px{suffix}")
    if "student_px" in test:
        full = test["student_px"]
        print("\n  paired contribution (positive means the modality helps)")
        if "without_emg_px" in test:
            print(f"    remove EMG : {test['without_emg_px'] - full:+.1f} px")
        if "shuffled_emg_px" in test:
            print(f"    shuffle EMG: {test['shuffled_emg_px'] - full:+.1f} px")
        if "without_imu_px" in test:
            print(f"    remove IMU : {test['without_imu_px'] - full:+.1f} px")
    print("\n  student error by lead")
    for lead in evaluation_leads:
        key = f"student_lead_{lead}_px"
        if key in test:
            print(f"    {1000.0 * lead / rate:5.0f} ms: {test[key]:7.1f} px")
    print(
        f"\n  student trajectory: {test.get('student_trajectory_cm', float('nan')):.2f} cm"
        f" | IMU-only tracking: "
        f"{test.get('imu_tracking_trajectory_cm', float('nan')):.2f} cm"
        f" | latent RMSE: {test.get('latent_rmse', float('nan')):.4f}"
    )

    save_json({"config": config, "history": history, "test": test}, output / "results.json")
    torch.save(
        {"model_state": clone_state(model), "config": config, "test": test},
        output / "final.pt",
    )
    print(f"wrote {output / 'results.json'} and {output / 'final.pt'}")


if __name__ == "__main__":
    main()
