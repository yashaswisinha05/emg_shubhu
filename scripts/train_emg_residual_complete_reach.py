#!/usr/bin/env python3
"""Train temporal EMG to correct the 3D residual left by the current model.

This isolated successor loads a trained soft-routed checkpoint, preserves its
screen and IMU-base predictions, and adds per-path queries that attend to
causal EMG tokens. Stage 1 freezes the loaded model and trains only the
zero-initialized residual branch. Stage 2 jointly fine-tunes at a reduced
learning rate. VIVE remains a label and privileged-teacher input only.

Example::

    python scripts/train_emg_residual_complete_reach.py \
      --root "/media/.../emg_imu_vive" \
      --initial-checkpoint runs/soft_routed_complete_reach/final.pt \
      --config configs/tracked_emg_residual_complete_reach.yaml \
      --cache-dir artifacts/tracked_cache_posture --device cuda \
      --epochs 30 --lead-window-ms 0 400 \
      --output-dir runs/emg_residual_complete_reach
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import train_channel_horizon_distillation_model as channel  # noqa: E402
from scripts import train_complete_reach_model as complete  # noqa: E402
from scripts import train_latent_distillation_model as base  # noqa: E402
from scripts import train_soft_routed_complete_reach as soft  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    emg_feature_count,
    imu_feature_count,
)
from emg_touch.models.emg_residual_complete_reach import (  # noqa: E402
    EMGResidualCompleteReachModel,
)


_ORIGINAL_TRAIN_STUDENT_PHASE = base.train_student_phase
_ORIGINAL_SET_TRAINABLE = base.set_trainable
_INITIAL_CHECKPOINT: Path | None = None


def residual_losses(
    outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Supervise the new branch on error left by the frozen base path."""
    settings = config["model"]["emg_temporal_residual"]
    scale = max(float(config["model"].get("trajectory_limit_m", 0.8)), 1e-6)
    beta = float(settings.get("residual_huber_beta", 0.05))
    path_target = (
        window["trajectory_target"]
        - outputs["pre_emg_residual_trajectory"].detach()
    )
    endpoint_target = (
        window["endpoint_3d_target"]
        - outputs["pre_emg_residual_endpoint"].detach()
    )
    path = F.smooth_l1_loss(
        outputs["emg_temporal_residual"] / scale,
        path_target / scale,
        beta=beta,
    )
    endpoint = F.smooth_l1_loss(
        outputs["emg_temporal_endpoint_residual"] / scale,
        endpoint_target / scale,
        beta=beta,
    )
    regularization = (
        outputs["emg_temporal_raw_path_residual"].square().mean()
        + outputs["emg_temporal_raw_endpoint_residual"].square().mean()
    )
    return {
        "emg_residual_path": path,
        "emg_residual_endpoint": endpoint,
        "emg_residual_regularization": regularization,
    }


def student_objective(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    combined = soft.student_objective(
        outputs, teacher_outputs, window, config
    )
    settings = config["model"]["emg_temporal_residual"]
    fused = residual_losses(outputs, window, config)
    emg = residual_losses(outputs["emg_only"], window, config)
    combined["loss"] = (
        combined["loss"]
        + float(settings.get("residual_weight", 0.50))
        * fused["emg_residual_path"]
        + float(settings.get("endpoint_residual_weight", 0.50))
        * fused["emg_residual_endpoint"]
        + float(settings.get("emg_only_residual_weight", 0.25))
        * (
            emg["emg_residual_path"]
            + emg["emg_residual_endpoint"]
        )
        + float(settings.get("residual_regularization_weight", 0.001))
        * fused["emg_residual_regularization"]
    )
    combined.update({name: value.detach() for name, value in fused.items()})
    combined.update({
        f"emg_only_{name}": value.detach() for name, value in emg.items()
    })
    combined["emg_temporal_path_gate"] = outputs[
        "emg_temporal_path_gate"
    ].mean().detach()
    combined["emg_temporal_endpoint_gate"] = outputs[
        "emg_temporal_endpoint_gate"
    ].mean().detach()
    return combined


def load_initial_checkpoint(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Replace teacher retraining with the complete successful checkpoint."""
    model = args[0] if args else kwargs["model"]
    output = args[11] if len(args) > 11 else kwargs["output"]
    if _INITIAL_CHECKPOINT is None:
        raise RuntimeError("initial checkpoint was not configured")
    payload = torch.load(_INITIAL_CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(
            f"{_INITIAL_CHECKPOINT} must contain a model_state dictionary"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_prefix = "student.emg_temporal_residual_head."
    invalid_missing = [key for key in missing if not key.startswith(allowed_prefix)]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "initial checkpoint is not compatible with the soft-routed base; "
            f"missing={invalid_missing}, unexpected={list(unexpected)}"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": base.clone_state(model),
            "source_checkpoint": str(_INITIAL_CHECKPOINT),
            "missing_new_parameters": list(missing),
        },
        output / "initialized.pt",
    )
    print(
        f"initialized from {_INITIAL_CHECKPOINT}; retained {len(state)} tensors; "
        f"created {len(missing)} zero-safe EMG residual tensors"
    )
    return [{
        "phase": "initialization",
        "source_checkpoint": str(_INITIAL_CHECKPOINT),
        "new_parameter_tensors": len(missing),
    }]


def staged_train_student_phase(
    phase: str,
    model: EMGResidualCompleteReachModel,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
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
    if phase != "student" or unfreeze_decoder:
        return _ORIGINAL_TRAIN_STUDENT_PHASE(
            phase, model, train_loader, validation_loader, config, epochs,
            context_samples, patch_length, lead_window, evaluation_leads,
            canvas_tensor, mean_target, device, output, difficulty, adaptive,
            unfreeze_decoder,
        )
    settings = config["model"]["emg_temporal_residual"]
    warmup_epochs = int(settings.get("warmup_epochs", 10))

    def residual_only(module: torch.nn.Module, enabled: bool) -> None:
        if module is model.student and enabled:
            _ORIGINAL_SET_TRAINABLE(module, False)
            _ORIGINAL_SET_TRAINABLE(
                model.student.emg_temporal_residual_head, True
            )
        elif module is model.guidance and enabled:
            _ORIGINAL_SET_TRAINABLE(module, False)
        else:
            _ORIGINAL_SET_TRAINABLE(module, enabled)

    model.residual_warmup = True
    base.set_trainable = residual_only
    try:
        warm_history, _, _ = _ORIGINAL_TRAIN_STUDENT_PHASE(
            "emg_residual_warmup",
            model,
            train_loader,
            validation_loader,
            config,
            warmup_epochs,
            context_samples,
            patch_length,
            lead_window,
            evaluation_leads,
            canvas_tensor,
            mean_target,
            device,
            output,
            difficulty,
            adaptive,
            False,
        )
    finally:
        base.set_trainable = _ORIGINAL_SET_TRAINABLE
        model.residual_warmup = False

    original_learning_rate = float(config["training"]["learning_rate"])
    factor = float(settings.get("joint_learning_rate_factor", 0.10))
    config["training"]["learning_rate"] = original_learning_rate * factor
    try:
        joint_history, best_state, best = _ORIGINAL_TRAIN_STUDENT_PHASE(
            "joint_finetune",
            model,
            train_loader,
            validation_loader,
            config,
            epochs,
            context_samples,
            patch_length,
            lead_window,
            evaluation_leads,
            canvas_tensor,
            mean_target,
            device,
            output,
            difficulty,
            adaptive,
            False,
        )
    finally:
        config["training"]["learning_rate"] = original_learning_rate
    return warm_history + joint_history, best_state, best


def _extend(
    totals: dict[str, list[float]], name: str, values: torch.Tensor
) -> None:
    totals.setdefault(name, []).extend(
        values.detach().cpu().reshape(-1).tolist()
    )


@torch.no_grad()
def evaluate_3d_emg_interventions(
    model: EMGResidualCompleteReachModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    device: torch.device,
) -> dict[str, float]:
    """Paired path, endpoint, and direction ablations for EMG in 3D."""
    model.eval()
    steps = int(config["model"]["teacher_trajectory_steps"])
    limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    generator = np.random.default_rng(0)
    totals: dict[str, list[float]] = {}

    for raw_batch in loader:
        if raw_batch is None:
            continue
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        for lead in evaluation_leads:
            window = complete.make_complete_reach_window(
                batch,
                context_samples,
                patch_length,
                steps,
                generator,
                limit,
                velocity_scale,
                fallback_canvas=None,
                fixed_lead=lead,
            )
            if window is None:
                continue
            conditions = {
                "full": model.student_forward(
                    window["emg"], window["imu"], window["time_mask"]
                ),
                "without_emg": model.student_forward(
                    torch.zeros_like(window["emg"]),
                    window["imu"],
                    window["time_mask"],
                ),
                "without_imu": model.student_forward(
                    window["emg"],
                    torch.zeros_like(window["imu"]),
                    window["time_mask"],
                ),
            }
            if window["emg"].size(0) > 1:
                conditions["shuffled_emg"] = model.student_forward(
                    torch.roll(window["emg"], shifts=1, dims=0),
                    window["imu"],
                    window["time_mask"],
                )
            target_path = window["trajectory_target"]
            target_endpoint = window["endpoint_3d_target"]
            for name, outputs in conditions.items():
                path = 100.0 * torch.linalg.vector_norm(
                    outputs["trajectory"] - target_path, dim=-1
                ).mean(dim=-1)
                endpoint = 100.0 * torch.linalg.vector_norm(
                    outputs["endpoint_3d"] - target_endpoint, dim=-1
                )
                cosine = F.cosine_similarity(
                    outputs["endpoint_3d"], target_endpoint, dim=-1, eps=1e-6
                ).clamp(-1.0, 1.0)
                angle = torch.rad2deg(torch.acos(cosine))
                _extend(totals, f"emg3d_{name}_path_cm", path)
                _extend(totals, f"emg3d_{name}_endpoint_cm", endpoint)
                _extend(totals, f"emg3d_{name}_angle_deg", angle)
                _extend(
                    totals,
                    f"emg3d_{name}_wrong_way",
                    (angle > 90.0).to(angle.dtype),
                )
            full = conditions["full"]
            _extend(
                totals,
                "emg3d_residual_magnitude_cm",
                100.0 * torch.linalg.vector_norm(
                    full["emg_temporal_residual"], dim=-1
                ).mean(dim=-1),
            )
            _extend(
                totals,
                "emg3d_path_gate",
                full["emg_temporal_path_gate"].mean(dim=-1),
            )

    metrics = {
        name: float(np.mean(values))
        for name, values in totals.items()
        if values
    }
    full_path = metrics.get("emg3d_full_path_cm", float("nan"))
    full_endpoint = metrics.get("emg3d_full_endpoint_cm", float("nan"))
    for condition in ("without_emg", "shuffled_emg", "without_imu"):
        path_key = f"emg3d_{condition}_path_cm"
        endpoint_key = f"emg3d_{condition}_endpoint_cm"
        if path_key in metrics:
            metrics[f"emg3d_{condition}_path_cost_cm"] = (
                metrics[path_key] - full_path
            )
        if endpoint_key in metrics:
            metrics[f"emg3d_{condition}_endpoint_cost_cm"] = (
                metrics[endpoint_key] - full_endpoint
            )
    return metrics


def _has_option(name: str) -> bool:
    return any(
        value == name or value.startswith(f"{name}=")
        for value in sys.argv[1:]
    )


def _option_value(name: str, default: str) -> str:
    arguments = sys.argv[1:]
    for index, value in enumerate(arguments):
        if value.startswith(f"{name}="):
            return value.split("=", 1)[1]
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return default


def _pop_option(name: str, default: str) -> str:
    arguments = sys.argv[1:]
    for index, value in enumerate(arguments):
        if value.startswith(f"{name}="):
            del sys.argv[index + 1]
            return value.split("=", 1)[1]
        if value == name and index + 1 < len(arguments):
            result = arguments[index + 1]
            del sys.argv[index + 1 : index + 3]
            return result
    return default


def postprocess_3d_interventions(
    output: Path, root: str, cache_dir: str, device_name: str
) -> None:
    final_path = output / "final.pt"
    payload = torch.load(final_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    device = base.choose_device(device_name)
    loaders = base.build_experiment_loaders(config, root, Path(cache_dir))
    test_loader = loaders[2]
    model = EMGResidualCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    rate = base.effective_rate(config)
    context_samples = max(
        int(config["model"]["patch_length"]),
        base.milliseconds_to_samples(
            float(config["model"].get("context_ms", 2000.0)), rate
        ),
    )
    leads = tuple(dict.fromkeys(
        base.milliseconds_to_samples(value, rate)
        for value in config["distillation"].get(
            "evaluation_leads_ms", [0, 50, 100, 200, 300, 400]
        )
    ))
    metrics = evaluate_3d_emg_interventions(
        model,
        test_loader,
        config,
        context_samples,
        int(config["model"]["patch_length"]),
        leads,
        device,
    )
    payload.setdefault("test", {}).update(metrics)
    torch.save(payload, final_path)
    results_path = output / "results.json"
    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    results.setdefault("test", {}).update(metrics)
    base.save_json(results, results_path)

    print("\n=== paired EMG contribution to complete 3D motion ===")
    for condition, label in (
        ("full", "full EMG+IMU"),
        ("without_emg", "without EMG"),
        ("shuffled_emg", "shuffled EMG"),
        ("without_imu", "without IMU"),
    ):
        path_key = f"emg3d_{condition}_path_cm"
        if path_key not in metrics:
            continue
        print(
            f"  {label:13}: path={metrics[path_key]:6.2f}cm | "
            f"endpoint={metrics[f'emg3d_{condition}_endpoint_cm']:6.2f}cm | "
            f"angle={metrics[f'emg3d_{condition}_angle_deg']:6.1f}° | "
            f"wrong-way={100.0 * metrics[f'emg3d_{condition}_wrong_way']:5.1f}%"
        )
    print("  positive paired cost means the removed/shuffled modality helped")
    print(
        f"  remove EMG path cost : "
        f"{metrics.get('emg3d_without_emg_path_cost_cm', float('nan')):+.2f}cm"
    )
    print(
        f"  shuffle EMG path cost: "
        f"{metrics.get('emg3d_shuffled_emg_path_cost_cm', float('nan')):+.2f}cm"
    )
    print(
        f"  learned residual magnitude: "
        f"{metrics.get('emg3d_residual_magnitude_cm', float('nan')):.2f}cm | "
        f"mean gate={metrics.get('emg3d_path_gate', float('nan')):.3f}"
    )
    print(f"updated {results_path} and {final_path}")


def main() -> None:
    global _INITIAL_CHECKPOINT
    _INITIAL_CHECKPOINT = Path(_pop_option(
        "--initial-checkpoint", "runs/soft_routed_complete_reach/final.pt"
    ))
    help_requested = any(value in {"-h", "--help"} for value in sys.argv[1:])
    if not help_requested and not _INITIAL_CHECKPOINT.is_file():
        raise SystemExit(f"initial checkpoint not found: {_INITIAL_CHECKPOINT}")
    if not _has_option("--config"):
        sys.argv[1:1] = [
            "--config", "configs/tracked_emg_residual_complete_reach.yaml"
        ]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = [
            "--output-dir", "runs/emg_residual_complete_reach"
        ]

    output = Path(_option_value(
        "--output-dir", "runs/emg_residual_complete_reach"
    ))
    root = _option_value("--root", "")
    cache_dir = _option_value("--cache-dir", "artifacts/tracked_cache_posture")
    device_name = _option_value("--device", "cuda")

    base.make_distillation_window = complete.make_complete_reach_window
    base.milliseconds_to_samples = complete.milliseconds_to_samples
    base.evaluate = soft.evaluate
    base.train_teacher = load_initial_checkpoint
    base.train_student_phase = staged_train_student_phase
    channel.ChannelHorizonLatentDistillationModel = EMGResidualCompleteReachModel
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    print(
        "EMG residual reach: load soft-routed checkpoint -> freeze base and "
        "learn temporal EMG correction -> low-LR joint fine-tuning"
    )
    channel.main()
    postprocess_3d_interventions(output, root, cache_dir, device_name)


if __name__ == "__main__":
    main()
