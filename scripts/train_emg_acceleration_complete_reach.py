#!/usr/bin/env python3
"""Train an EMG-driven acceleration correction for complete 3D reaches.

VIVE position and velocity are supervision labels only.  The deployable model
still receives just causal EMG+IMU.  A zero-initialized EMG head predicts an
acceleration residual, integrates it twice, and adds the resulting path to the
previous best temporal-EMG-residual model.

Example::

    python scripts/train_emg_acceleration_complete_reach.py \
      --root "/media/.../emg_imu_vive" \
      --initial-checkpoint runs/emg_residual_complete_reach/final.pt \
      --config configs/tracked_emg_acceleration_complete_reach.yaml \
      --cache-dir artifacts/tracked_cache_posture --device cuda \
      --epochs 30 --lead-window-ms 0 400 \
      --output-dir runs/emg_acceleration_complete_reach
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
from scripts import train_emg_residual_complete_reach as residual  # noqa: E402
from scripts import train_latent_distillation_model as base  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    emg_feature_count,
    imu_feature_count,
)
from emg_touch.models.emg_acceleration_complete_reach import (  # noqa: E402
    EMGAccelerationCompleteReachModel,
    finite_difference,
    trajectory_kinematics,
)


_ORIGINAL_TRAIN_STUDENT_PHASE = base.train_student_phase
_ORIGINAL_SET_TRAINABLE = base.set_trainable
_INITIAL_CHECKPOINT: Path | None = None


def _true_duration(window: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    rate = max(base.effective_rate(config), 1e-6)
    return window["movement_samples"].to(torch.float32) / rate


def _smooth_velocity(velocity: torch.Tensor) -> torch.Tensor:
    # Suppress one-sample tracker jitter before differentiating. This operation
    # constructs a label; it is not an input transformation or test-time leak.
    if velocity.size(1) < 3:
        return velocity
    values = velocity.transpose(1, 2)
    values = F.pad(values, (1, 1), mode="replicate")
    values = F.avg_pool1d(values, kernel_size=3, stride=1)
    return values.transpose(1, 2)


def acceleration_losses(
    outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Velocity, acceleration, duration, and residual-dynamics supervision."""
    settings = config["model"]["emg_acceleration_dynamics"]
    duration = _true_duration(window, config)
    target_velocity = _smooth_velocity(window["velocity_target"])
    target_acceleration = finite_difference(target_velocity, duration)
    predicted_velocity, predicted_acceleration = trajectory_kinematics(
        outputs["trajectory"], duration
    )
    _, base_acceleration = trajectory_kinematics(
        outputs["pre_acceleration_trajectory"].detach(), duration
    )
    acceleration_scale = max(
        float(settings.get("acceleration_loss_scale_mps2", 2.0)), 1e-6
    )
    velocity_scale = max(
        float(config["model"].get("velocity_scale_mps", 1.0)), 1e-6
    )
    beta = float(settings.get("huber_beta", 0.05))
    acceleration = F.smooth_l1_loss(
        predicted_acceleration / acceleration_scale,
        target_acceleration / acceleration_scale,
        beta=beta,
    )
    velocity = F.smooth_l1_loss(
        predicted_velocity / velocity_scale,
        target_velocity / velocity_scale,
        beta=beta,
    )
    residual_target = target_acceleration - base_acceleration
    residual_fit = F.smooth_l1_loss(
        outputs["emg_acceleration_residual"] / acceleration_scale,
        residual_target / acceleration_scale,
        beta=beta,
    )
    target_jerk = finite_difference(target_acceleration, duration)
    predicted_jerk = finite_difference(predicted_acceleration, duration)
    jerk_scale = max(float(settings.get("jerk_loss_scale_mps3", 10.0)), 1e-6)
    jerk = F.smooth_l1_loss(
        predicted_jerk / jerk_scale,
        target_jerk / jerk_scale,
        beta=beta,
    )
    duration_loss = F.smooth_l1_loss(
        outputs["predicted_movement_duration_s"], duration, beta=0.05
    )
    regularization = outputs["emg_acceleration_raw"].square().mean()
    return {
        "dynamics_acceleration": acceleration,
        "dynamics_velocity": velocity,
        "dynamics_residual_fit": residual_fit,
        "dynamics_jerk": jerk,
        "dynamics_duration": duration_loss,
        "dynamics_regularization": regularization,
    }


def student_objective(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    combined = residual.student_objective(
        outputs, teacher_outputs, window, config
    )
    settings = config["model"]["emg_acceleration_dynamics"]
    fused = acceleration_losses(outputs, window, config)
    emg = acceleration_losses(outputs["emg_only"], window, config)
    combined["loss"] = (
        combined["loss"]
        + float(settings.get("acceleration_weight", 0.20))
        * fused["dynamics_acceleration"]
        + float(settings.get("velocity_weight", 0.10))
        * fused["dynamics_velocity"]
        + float(settings.get("residual_fit_weight", 0.35))
        * fused["dynamics_residual_fit"]
        + float(settings.get("jerk_weight", 0.025))
        * fused["dynamics_jerk"]
        + float(settings.get("duration_weight", 0.05))
        * fused["dynamics_duration"]
        + float(settings.get("emg_only_acceleration_weight", 0.10))
        * (
            emg["dynamics_acceleration"]
            + emg["dynamics_residual_fit"]
        )
        + float(settings.get("regularization_weight", 0.0005))
        * fused["dynamics_regularization"]
    )
    combined.update({name: value.detach() for name, value in fused.items()})
    combined.update({
        f"emg_only_{name}": value.detach() for name, value in emg.items()
    })
    combined["emg_acceleration_gate"] = outputs[
        "emg_acceleration_gate"
    ].mean().detach()
    return combined


def load_initial_checkpoint(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    model = args[0] if args else kwargs["model"]
    output = args[11] if len(args) > 11 else kwargs["output"]
    if _INITIAL_CHECKPOINT is None:
        raise RuntimeError("initial checkpoint was not configured")
    payload = torch.load(_INITIAL_CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"{_INITIAL_CHECKPOINT} must contain model_state")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = "student.emg_acceleration_dynamics_head."
    invalid = [key for key in missing if not key.startswith(allowed)]
    if invalid or unexpected:
        raise RuntimeError(
            "checkpoint is not compatible with the EMG-residual base; "
            f"missing={invalid}, unexpected={list(unexpected)}"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": base.clone_state(model),
        "source_checkpoint": str(_INITIAL_CHECKPOINT),
        "missing_new_parameters": list(missing),
    }, output / "initialized.pt")
    print(
        f"initialized from {_INITIAL_CHECKPOINT}; created {len(missing)} "
        "zero-safe acceleration-dynamics tensors"
    )
    return [{
        "phase": "initialization",
        "source_checkpoint": str(_INITIAL_CHECKPOINT),
        "new_parameter_tensors": len(missing),
    }]


def staged_train_student_phase(
    phase: str,
    model: EMGAccelerationCompleteReachModel,
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
    settings = config["model"]["emg_acceleration_dynamics"]
    warmup_epochs = int(settings.get("warmup_epochs", 10))

    def acceleration_only(module: torch.nn.Module, enabled: bool) -> None:
        if module is model.student and enabled:
            _ORIGINAL_SET_TRAINABLE(module, False)
            _ORIGINAL_SET_TRAINABLE(
                model.student.emg_acceleration_dynamics_head, True
            )
        elif module is model.guidance and enabled:
            _ORIGINAL_SET_TRAINABLE(module, False)
        else:
            _ORIGINAL_SET_TRAINABLE(module, enabled)

    model.acceleration_warmup = True
    base.set_trainable = acceleration_only
    try:
        warm_history, _, _ = _ORIGINAL_TRAIN_STUDENT_PHASE(
            "acceleration_warmup", model, train_loader, validation_loader,
            config, warmup_epochs, context_samples, patch_length, lead_window,
            evaluation_leads, canvas_tensor, mean_target, device, output,
            difficulty, adaptive, False,
        )
    finally:
        base.set_trainable = _ORIGINAL_SET_TRAINABLE
        model.acceleration_warmup = False

    original_lr = float(config["training"]["learning_rate"])
    config["training"]["learning_rate"] = original_lr * float(
        settings.get("joint_learning_rate_factor", 0.05)
    )
    try:
        joint_history, best_state, best = _ORIGINAL_TRAIN_STUDENT_PHASE(
            "acceleration_joint_finetune", model, train_loader,
            validation_loader, config, epochs, context_samples, patch_length,
            lead_window, evaluation_leads, canvas_tensor, mean_target, device,
            output, difficulty, adaptive, False,
        )
    finally:
        config["training"]["learning_rate"] = original_lr
    return warm_history + joint_history, best_state, best


@torch.no_grad()
def evaluate_dynamics(
    model: EMGAccelerationCompleteReachModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    device: torch.device,
) -> dict[str, float]:
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
                batch, context_samples, patch_length, steps, generator, limit,
                velocity_scale, fixed_lead=lead,
            )
            if window is None:
                continue
            outputs = model.student_forward(
                window["emg"], window["imu"], window["time_mask"]
            )
            duration = _true_duration(window, config)
            target_velocity = _smooth_velocity(window["velocity_target"])
            target_acceleration = finite_difference(target_velocity, duration)
            _, predicted_acceleration = trajectory_kinematics(
                outputs["trajectory"], duration
            )
            final_path = 100.0 * torch.linalg.vector_norm(
                outputs["trajectory"] - window["trajectory_target"], dim=-1
            ).mean(dim=-1)
            before_path = 100.0 * torch.linalg.vector_norm(
                outputs["pre_acceleration_trajectory"]
                - window["trajectory_target"], dim=-1
            ).mean(dim=-1)
            final_endpoint = 100.0 * torch.linalg.vector_norm(
                outputs["endpoint_3d"] - window["endpoint_3d_target"], dim=-1
            )
            before_endpoint = 100.0 * torch.linalg.vector_norm(
                outputs["pre_acceleration_endpoint"]
                - window["endpoint_3d_target"], dim=-1
            )
            acceleration_rmse = torch.sqrt(
                (predicted_acceleration - target_acceleration).square().mean(
                    dim=(1, 2)
                )
            )
            values = {
                "dynamics_path_cm": final_path,
                "dynamics_path_before_cm": before_path,
                "dynamics_endpoint_cm": final_endpoint,
                "dynamics_endpoint_before_cm": before_endpoint,
                "dynamics_acceleration_rmse_mps2": acceleration_rmse,
                "dynamics_duration_s": outputs["predicted_movement_duration_s"],
                "dynamics_duration_target_s": duration,
                "dynamics_gate": outputs["emg_acceleration_gate"].mean(dim=-1),
                "dynamics_correction_cm": 100.0 * torch.linalg.vector_norm(
                    outputs["emg_integrated_position_residual"], dim=-1
                ).mean(dim=-1),
            }
            for name, tensor in values.items():
                totals.setdefault(name, []).extend(
                    tensor.detach().cpu().reshape(-1).tolist()
                )
    metrics = {name: float(np.mean(values)) for name, values in totals.items()}
    if metrics:
        metrics["dynamics_path_gain_cm"] = (
            metrics["dynamics_path_before_cm"] - metrics["dynamics_path_cm"]
        )
        metrics["dynamics_endpoint_gain_cm"] = (
            metrics["dynamics_endpoint_before_cm"]
            - metrics["dynamics_endpoint_cm"]
        )
        metrics["dynamics_duration_mae_s"] = float(np.mean(np.abs(
            np.asarray(totals["dynamics_duration_s"])
            - np.asarray(totals["dynamics_duration_target_s"])
        )))
    return metrics


def postprocess(output: Path, root: str, cache_dir: str, device_name: str) -> None:
    final_path = output / "final.pt"
    payload = torch.load(final_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    device = base.choose_device(device_name)
    test_loader = base.build_experiment_loaders(
        config, root, Path(cache_dir)
    )[2]
    model = EMGAccelerationCompleteReachModel(
        config, emg_feature_count(config["data"]), imu_feature_count(config["data"])
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
    emg_metrics = residual.evaluate_3d_emg_interventions(
        model, test_loader, config, context_samples,
        int(config["model"]["patch_length"]), leads, device,
    )
    dynamics_metrics = evaluate_dynamics(
        model, test_loader, config, context_samples,
        int(config["model"]["patch_length"]), leads, device,
    )
    metrics = {**emg_metrics, **dynamics_metrics}
    payload.setdefault("test", {}).update(metrics)
    torch.save(payload, final_path)
    results_path = output / "results.json"
    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    results.setdefault("test", {}).update(metrics)
    base.save_json(results, results_path)
    print("\n=== EMG acceleration dynamics test ===")
    print(
        f"  path: {metrics.get('dynamics_path_cm', float('nan')):.2f}cm | "
        f"before dynamics={metrics.get('dynamics_path_before_cm', float('nan')):.2f}cm | "
        f"gain={metrics.get('dynamics_path_gain_cm', float('nan')):+.2f}cm"
    )
    print(
        f"  endpoint: {metrics.get('dynamics_endpoint_cm', float('nan')):.2f}cm | "
        f"gain={metrics.get('dynamics_endpoint_gain_cm', float('nan')):+.2f}cm | "
        f"acceleration RMSE={metrics.get('dynamics_acceleration_rmse_mps2', float('nan')):.2f}m/s²"
    )
    print(
        f"  duration MAE={1000.0 * metrics.get('dynamics_duration_mae_s', float('nan')):.1f}ms | "
        f"correction={metrics.get('dynamics_correction_cm', float('nan')):.2f}cm | "
        f"gate={metrics.get('dynamics_gate', float('nan')):.3f}"
    )
    print(
        f"  remove EMG path cost="
        f"{metrics.get('emg3d_without_emg_path_cost_cm', float('nan')):+.2f}cm | "
        f"shuffle EMG path cost="
        f"{metrics.get('emg3d_shuffled_emg_path_cost_cm', float('nan')):+.2f}cm"
    )
    print(f"updated {results_path} and {final_path}")


def _has_option(name: str) -> bool:
    return any(
        value == name or value.startswith(f"{name}=") for value in sys.argv[1:]
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
            del sys.argv[index + 1:index + 3]
            return result
    return default


def main() -> None:
    global _INITIAL_CHECKPOINT
    _INITIAL_CHECKPOINT = Path(_pop_option(
        "--initial-checkpoint", "runs/emg_residual_complete_reach/final.pt"
    ))
    help_requested = any(value in {"-h", "--help"} for value in sys.argv[1:])
    if not help_requested and not _INITIAL_CHECKPOINT.is_file():
        raise SystemExit(f"initial checkpoint not found: {_INITIAL_CHECKPOINT}")
    if not _has_option("--config"):
        sys.argv[1:1] = [
            "--config", "configs/tracked_emg_acceleration_complete_reach.yaml"
        ]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = [
            "--output-dir", "runs/emg_acceleration_complete_reach"
        ]
    output = Path(_option_value(
        "--output-dir", "runs/emg_acceleration_complete_reach"
    ))
    root = _option_value("--root", "")
    cache_dir = _option_value("--cache-dir", "artifacts/tracked_cache_posture")
    device_name = _option_value("--device", "cuda")

    base.make_distillation_window = complete.make_complete_reach_window
    base.milliseconds_to_samples = complete.milliseconds_to_samples
    base.evaluate = residual.soft.evaluate
    base.train_teacher = load_initial_checkpoint
    base.train_student_phase = staged_train_student_phase
    channel.ChannelHorizonLatentDistillationModel = (
        EMGAccelerationCompleteReachModel
    )
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    print(
        "EMG acceleration reach: load EMG-residual checkpoint -> learn a "
        "velocity/acceleration-supervised integrated correction -> low-LR joint fit"
    )
    channel.main()
    postprocess(output, root, cache_dir, device_name)


if __name__ == "__main__":
    main()
