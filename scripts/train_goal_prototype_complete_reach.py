#!/usr/bin/env python3
"""Train one wearable goal distribution for screen and complete 3D reach.

This isolated experiment loads the successful temporal-EMG-residual
checkpoint.  Its existing 8x5 screen heatmap becomes a soft target
distribution shared with a bank of training-only 3D residual prototypes.
The predicted 3D endpoint also supplies a bounded residual correction to the
screen point.  Every new output is zero-initialized, so initialization exactly
reproduces the source checkpoint.

VIVE supplies supervision labels and the already trained privileged teacher
only.  Deployment remains ``student_forward(emg, imu, time_mask)``.
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
from scripts import train_soft_routed_complete_reach as soft  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    emg_feature_count,
    imu_feature_count,
)
from emg_touch.models.goal_prototype_complete_reach import (  # noqa: E402
    GoalPrototypeCompleteReachModel,
)


_ORIGINAL_TRAIN_STUDENT_PHASE = base.train_student_phase
_ORIGINAL_SET_TRAINABLE = base.set_trainable
_INITIAL_CHECKPOINT: Path | None = None


def _smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: float,
    beta: float,
) -> torch.Tensor:
    return F.smooth_l1_loss(
        prediction / max(float(scale), 1e-6),
        target / max(float(scale), 1e-6),
        beta=float(beta),
    )


def goal_bridge_losses(
    outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Learn per-target residual prototypes and bidirectional task coupling."""
    settings = config["model"].get("goal_prototype_bridge", {})
    scale = float(config["model"].get("trajectory_limit_m", 0.8))
    beta = float(settings.get("huber_beta", 0.05))
    grid_width, grid_height = map(
        int, config["model"].get("grid_size", [8, 5])
    )
    target_class = base.target_cell_labels(
        window["target"], grid_width, grid_height
    )
    rows = torch.arange(target_class.size(0), device=target_class.device)
    oracle_path = outputs["goal_all_path_prototypes"][target_class]
    oracle_endpoint = outputs["goal_all_endpoint_prototypes"][target_class]
    target_path_residual = (
        window["trajectory_target"] - outputs["pre_goal_trajectory"].detach()
    )
    target_endpoint_residual = (
        window["endpoint_3d_target"] - outputs["pre_goal_endpoint"].detach()
    )
    target_screen_delta = (
        window["target"] - outputs["pre_goal_prediction"].detach()
    )

    oracle_path_loss = _smooth_l1(
        oracle_path, target_path_residual, scale, beta
    )
    oracle_endpoint_loss = _smooth_l1(
        oracle_endpoint, target_endpoint_residual, scale, beta
    )
    geometry = F.smooth_l1_loss(
        outputs["goal_geometry_delta"],
        target_screen_delta,
        beta=float(settings.get("screen_huber_beta", 0.05)),
    )
    goal_classification = F.nll_loss(
        outputs["goal_probabilities"].clamp_min(1e-8).log(), target_class
    )

    target_path = window["trajectory_target"]
    full_path_error = torch.linalg.vector_norm(
        outputs["trajectory"] - target_path, dim=-1
    ).mean(dim=-1)
    base_path_error = torch.linalg.vector_norm(
        outputs["pre_goal_trajectory"].detach() - target_path, dim=-1
    ).mean(dim=-1)
    target_endpoint = window["endpoint_3d_target"]
    full_endpoint_error = torch.linalg.vector_norm(
        outputs["endpoint_3d"] - target_endpoint, dim=-1
    )
    base_endpoint_error = torch.linalg.vector_norm(
        outputs["pre_goal_endpoint"].detach() - target_endpoint, dim=-1
    )
    canvas = window["canvas_size"]
    full_screen_error = torch.linalg.vector_norm(
        (outputs["prediction"] - window["target"]) * canvas, dim=-1
    )
    base_screen_error = torch.linalg.vector_norm(
        (outputs["pre_goal_prediction"].detach() - window["target"]) * canvas,
        dim=-1,
    )
    preservation = (
        F.relu(full_path_error - base_path_error).mean()
        + F.relu(full_endpoint_error - base_endpoint_error).mean()
        + F.relu((full_screen_error - base_screen_error) / 100.0).mean()
    )
    regularization = (
        outputs["goal_selected_path_prototype"].square().mean()
        + outputs["goal_selected_endpoint_prototype"].square().mean()
        + outputs["goal_geometry_delta"].square().mean()
    )
    # Useful diagnostic: probability assigned to the correct goal cell.
    correct_probability = outputs["goal_probabilities"][rows, target_class].mean()
    return {
        "goal_oracle_path": oracle_path_loss,
        "goal_oracle_endpoint": oracle_endpoint_loss,
        "goal_geometry_screen": geometry,
        "goal_classification": goal_classification,
        "goal_base_preservation": preservation,
        "goal_regularization": regularization,
        "goal_correct_probability": correct_probability,
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
    losses = goal_bridge_losses(outputs, window, config)
    settings = config["model"].get("goal_prototype_bridge", {})
    combined["loss"] = (
        combined["loss"]
        + float(settings.get("oracle_path_weight", 0.50))
        * losses["goal_oracle_path"]
        + float(settings.get("oracle_endpoint_weight", 0.50))
        * losses["goal_oracle_endpoint"]
        + float(settings.get("geometry_screen_weight", 0.50))
        * losses["goal_geometry_screen"]
        + float(settings.get("goal_classification_weight", 0.10))
        * losses["goal_classification"]
        + float(settings.get("base_preservation_weight", 0.25))
        * losses["goal_base_preservation"]
        + float(settings.get("regularization_weight", 0.001))
        * losses["goal_regularization"]
    )
    combined.update({name: value.detach() for name, value in losses.items()})
    for name in ("path", "endpoint", "geometry"):
        combined[f"goal_{name}_gate"] = outputs[
            f"goal_{name}_gate"
        ].mean().detach()
    return combined


def load_initial_checkpoint(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Load all successful weights and allow only the new goal bridge."""
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
    allowed = "student.goal_prototype_bridge."
    invalid_missing = [key for key in missing if not key.startswith(allowed)]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "initial checkpoint is not compatible with the EMG-residual base; "
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
        f"created {len(missing)} zero-safe goal-bridge tensors"
    )
    return [{
        "phase": "initialization",
        "source_checkpoint": str(_INITIAL_CHECKPOINT),
        "new_parameter_tensors": len(missing),
    }]


def staged_train_student_phase(
    phase: str,
    model: GoalPrototypeCompleteReachModel,
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
    settings = config["model"].get("goal_prototype_bridge", {})
    warmup_epochs = int(settings.get("warmup_epochs", 8))

    def goal_only(module: torch.nn.Module, enabled: bool) -> None:
        if module is model.student and enabled:
            _ORIGINAL_SET_TRAINABLE(module, False)
            _ORIGINAL_SET_TRAINABLE(model.student.goal_prototype_bridge, True)
        elif module is model.guidance and enabled:
            _ORIGINAL_SET_TRAINABLE(module, False)
        else:
            _ORIGINAL_SET_TRAINABLE(module, enabled)

    model.goal_warmup = True
    base.set_trainable = goal_only
    try:
        warm_history, _, _ = _ORIGINAL_TRAIN_STUDENT_PHASE(
            "goal_bridge_warmup", model, train_loader, validation_loader,
            config, warmup_epochs, context_samples, patch_length, lead_window,
            evaluation_leads, canvas_tensor, mean_target, device, output,
            difficulty, adaptive, False,
        )
    finally:
        base.set_trainable = _ORIGINAL_SET_TRAINABLE
        model.goal_warmup = False

    original_lr = float(config["training"]["learning_rate"])
    config["training"]["learning_rate"] = original_lr * float(
        settings.get("joint_learning_rate_factor", 0.10)
    )
    try:
        joint_history, best_state, best = _ORIGINAL_TRAIN_STUDENT_PHASE(
            "goal_joint_finetune", model, train_loader, validation_loader,
            config, epochs, context_samples, patch_length, lead_window,
            evaluation_leads, canvas_tensor, mean_target, device, output,
            difficulty, adaptive, False,
        )
    finally:
        config["training"]["learning_rate"] = original_lr
    return warm_history + joint_history, best_state, best


def _extend(store: dict[str, list[float]], name: str, value: torch.Tensor) -> None:
    store.setdefault(name, []).extend(value.detach().cpu().reshape(-1).tolist())


@torch.no_grad()
def evaluate_goal_bridge(
    model: GoalPrototypeCompleteReachModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    steps = int(config["model"]["teacher_trajectory_steps"])
    limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    grid_width, grid_height = map(
        int, config["model"].get("grid_size", [8, 5])
    )
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
                velocity_scale, canvas_tensor, fixed_lead=lead,
            )
            if window is None:
                continue
            outputs = model.student_forward(
                window["emg"], window["imu"], window["time_mask"]
            )
            target_class = base.target_cell_labels(
                window["target"], grid_width, grid_height
            )
            full_path = 100.0 * torch.linalg.vector_norm(
                outputs["trajectory"] - window["trajectory_target"], dim=-1
            ).mean(dim=-1)
            old_path = 100.0 * torch.linalg.vector_norm(
                outputs["pre_goal_trajectory"] - window["trajectory_target"],
                dim=-1,
            ).mean(dim=-1)
            full_endpoint = 100.0 * torch.linalg.vector_norm(
                outputs["endpoint_3d"] - window["endpoint_3d_target"], dim=-1
            )
            old_endpoint = 100.0 * torch.linalg.vector_norm(
                outputs["pre_goal_endpoint"] - window["endpoint_3d_target"],
                dim=-1,
            )
            full_screen = torch.linalg.vector_norm(
                (outputs["prediction"] - window["target"])
                * window["canvas_size"], dim=-1,
            )
            old_screen = torch.linalg.vector_norm(
                (outputs["pre_goal_prediction"] - window["target"])
                * window["canvas_size"], dim=-1,
            )
            for name, value in (
                ("goal_full_path_cm", full_path),
                ("goal_pre_bridge_path_cm", old_path),
                ("goal_full_endpoint_cm", full_endpoint),
                ("goal_pre_bridge_endpoint_cm", old_endpoint),
                ("goal_full_screen_px", full_screen),
                ("goal_pre_bridge_screen_px", old_screen),
                ("goal_target_accuracy", (outputs["goal_predicted_cell"] == target_class).float()),
                ("goal_path_gate", outputs["goal_path_gate"]),
                ("goal_endpoint_gate", outputs["goal_endpoint_gate"]),
                ("goal_geometry_gate", outputs["goal_geometry_gate"]),
            ):
                _extend(totals, name, value)
    metrics = {
        name: float(np.mean(values)) for name, values in totals.items() if values
    }
    for task_name, unit in (
        ("path", "cm"), ("endpoint", "cm"), ("screen", "px")
    ):
        full = metrics.get(f"goal_full_{task_name}_{unit}")
        old = metrics.get(f"goal_pre_bridge_{task_name}_{unit}")
        if full is not None and old is not None:
            metrics[f"goal_{task_name}_gain_{unit}"] = old - full
    return metrics


@torch.no_grad()
def evaluate(
    model: GoalPrototypeCompleteReachModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    mean_target: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    metrics = soft.evaluate(
        model, loader, config, context_samples, patch_length,
        evaluation_leads, canvas_tensor, mean_target, device,
    )
    metrics.update(evaluate_goal_bridge(
        model, loader, config, context_samples, patch_length,
        evaluation_leads, canvas_tensor, device,
    ))
    return metrics


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
            del sys.argv[index + 1 : index + 3]
            return result
    return default


def postprocess_interventions(
    output: Path, root: str, cache_dir: str, device_name: str
) -> None:
    final_path = output / "final.pt"
    payload = torch.load(final_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    device = base.choose_device(device_name)
    test_loader = base.build_experiment_loaders(
        config, root, Path(cache_dir)
    )[2]
    model = GoalPrototypeCompleteReachModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    rate = base.effective_rate(config)
    context_samples = max(
        int(config["model"]["patch_length"]),
        complete.milliseconds_to_samples(
            float(config["model"].get("context_ms", 2000.0)), rate
        ),
    )
    leads = tuple(dict.fromkeys(
        complete.milliseconds_to_samples(value, rate)
        for value in config["distillation"].get(
            "evaluation_leads_ms", [0, 50, 100, 200, 300, 400]
        )
    ))
    metrics = residual.evaluate_3d_emg_interventions(
        model, test_loader, config, context_samples,
        int(config["model"]["patch_length"]), leads, device,
    )
    payload.setdefault("test", {}).update(metrics)
    torch.save(payload, final_path)
    results_path = output / "results.json"
    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    results.setdefault("test", {}).update(metrics)
    base.save_json(results, results_path)
    test = results["test"]
    print("\n=== shared goal bridge test ===")
    print(
        f"  screen: {test.get('goal_full_screen_px', float('nan')):.1f}px | "
        f"before bridge={test.get('goal_pre_bridge_screen_px', float('nan')):.1f}px | "
        f"gain={test.get('goal_screen_gain_px', float('nan')):+.1f}px"
    )
    print(
        f"  3D path: {test.get('goal_full_path_cm', float('nan')):.2f}cm | "
        f"before bridge={test.get('goal_pre_bridge_path_cm', float('nan')):.2f}cm | "
        f"gain={test.get('goal_path_gain_cm', float('nan')):+.2f}cm"
    )
    print(
        f"  3D endpoint: {test.get('goal_full_endpoint_cm', float('nan')):.2f}cm | "
        f"gain={test.get('goal_endpoint_gain_cm', float('nan')):+.2f}cm | "
        f"goal-cell accuracy={100.0 * test.get('goal_target_accuracy', float('nan')):.1f}%"
    )
    print(
        f"  paired EMG path cost: remove="
        f"{test.get('emg3d_without_emg_path_cost_cm', float('nan')):+.2f}cm | "
        f"shuffle={test.get('emg3d_shuffled_emg_path_cost_cm', float('nan')):+.2f}cm"
    )
    print(f"updated {results_path} and {final_path}")


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
            "--config", "configs/tracked_goal_prototype_complete_reach.yaml"
        ]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = [
            "--output-dir", "runs/goal_prototype_complete_reach"
        ]
    output = Path(_option_value(
        "--output-dir", "runs/goal_prototype_complete_reach"
    ))
    root = _option_value("--root", "")
    cache_dir = _option_value("--cache-dir", "artifacts/tracked_cache_posture")
    device_name = _option_value("--device", "cuda")

    base.make_distillation_window = complete.make_complete_reach_window
    base.milliseconds_to_samples = complete.milliseconds_to_samples
    base.evaluate = evaluate
    base.train_teacher = load_initial_checkpoint
    base.train_student_phase = staged_train_student_phase
    channel.ChannelHorizonLatentDistillationModel = GoalPrototypeCompleteReachModel
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    print(
        "shared-goal reach: load EMG-residual checkpoint -> train target-grid "
        "3D residual prototypes + endpoint-to-screen correction -> low-LR joint fine-tune"
    )
    channel.main()
    postprocess_interventions(output, root, cache_dir, device_name)


if __name__ == "__main__":
    main()
