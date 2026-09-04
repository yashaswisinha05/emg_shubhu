#!/usr/bin/env python3
"""Train a causal wearable model to predict one complete reach.

At every observation from 400 ms before touch through touch, the deployable
student receives only the EMG+IMU prefix and predicts:

* final screen ``(x, y)``;
* onset-relative final 3D endpoint;
* the complete onset-to-touch 3D path;
* remaining time to touch.

The complete path target is stable across observation times.  VIVE constructs
privileged teacher inputs and labels only; it never enters ``student_forward``.
This experiment is isolated from the future-only rolling dual-head trainer.
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
from scripts import train_latent_distillation_model as base  # noqa: E402
from scripts import train_rolling_dual_head_model as rolling  # noqa: E402
from scripts import train_teacher_bridge_model as bridge  # noqa: E402
from emg_touch.models.complete_reach_distillation import (  # noqa: E402
    CompleteReachDistillationModel,
)


_ROLLING_STUDENT_OBJECTIVE = rolling.student_objective
_ROLLING_EVALUATE = rolling.evaluate


def milliseconds_to_samples(milliseconds: float, rate: float) -> int:
    """Permit a genuine zero-lead observation at touch."""
    return max(0, int(round(float(milliseconds) * float(rate) / 1000.0)))


def make_complete_reach_window(
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
    """Build causal prefixes with a cutoff-invariant full-reach target.

    The sample at the observation instant is included.  At zero lead this is
    the touch sample, so the model is explicitly trained on the complete
    wearable history.  VIVE positions from onset through touch are resampled
    to a fixed-length label relative to the onset position.
    """
    lengths = batch["lengths"]
    chosen: list[tuple[int, int, int, int]] = []
    for row in range(len(lengths)):
        touch = int(lengths[row]) - 1
        for _ in range(int(cutoffs_per_trial)):
            if fixed_lead is not None:
                lead = int(fixed_lead)
            elif lead_window is not None:
                low, high = sorted(map(int, lead_window))
                lead = int(generator.integers(low, high + 1))
            else:
                lead = int(generator.integers(0, max(1, touch + 1)))
            cut = touch - lead
            if cut < 0 or cut > touch:
                continue
            chosen.append((row, cut, touch, lead))
    if not chosen:
        return None

    prefix = max(int(context_samples), int(patch_length))
    device = batch["emg"].device
    emg_windows: list[torch.Tensor] = []
    imu_windows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    teacher_features: list[torch.Tensor] = []
    trajectory_targets: list[torch.Tensor] = []
    velocity_targets: list[torch.Tensor] = []
    endpoint_targets: list[torch.Tensor] = []
    current_position_targets: list[torch.Tensor] = []
    observed_fractions: list[float] = []
    movement_samples: list[int] = []
    source_paths: list[str] = []

    for row, cut, touch, _ in chosen:
        observed_end = cut + 1
        start = max(0, observed_end - prefix)
        available_count = observed_end - start
        mask = torch.zeros(prefix, dtype=torch.bool, device=device)
        mask[-available_count:] = True
        masks.append(mask)

        for source, destination in (
            (batch["emg"], emg_windows),
            (batch["imu"], imu_windows),
        ):
            available = source[row, start:observed_end]
            padded = source.new_zeros(prefix, source.size(-1))
            padded[-available.size(0) :] = available
            destination.append(padded)

        onset = int(batch["onset"][row])
        onset = min(max(onset, 0), max(0, touch - 1))
        indices = torch.linspace(
            onset, touch, int(teacher_steps), device=device
        ).round().long().clamp(onset, touch)
        origin = batch["position"][row, onset]
        full_position = (
            batch["position"][row].index_select(0, indices) - origin
        )
        full_velocity = batch["velocity"][row].index_select(0, indices)
        trajectory_targets.append(full_position)
        velocity_targets.append(full_velocity)
        endpoint_targets.append(batch["position"][row, touch] - origin)
        current_position_targets.append(batch["position"][row, cut] - origin)
        teacher_features.append(torch.cat([
            full_position / max(float(trajectory_limit_m), 1e-6),
            full_velocity / max(float(velocity_scale_mps), 1e-6),
        ], dim=-1))
        denominator = max(1, touch - onset)
        observed_fractions.append(
            float(min(max(cut - onset, 0), denominator)) / denominator
        )
        movement_samples.append(denominator)
        paths = batch.get("paths")
        source_paths.append(str(paths[row]) if paths is not None else str(row))

    rows = torch.tensor(
        [item[0] for item in chosen], dtype=torch.long, device=device
    )
    window: dict[str, Any] = {
        "emg": torch.stack(emg_windows),
        "imu": torch.stack(imu_windows),
        "time_mask": torch.stack(masks),
        "teacher_features": torch.stack(teacher_features),
        "trajectory_target": torch.stack(trajectory_targets),
        "complete_trajectory_target": torch.stack(trajectory_targets),
        # Additional supervision-only labels for dynamics successors. Existing
        # complete-reach models ignore these keys, so their behavior is unchanged.
        "velocity_target": torch.stack(velocity_targets),
        "movement_samples": torch.tensor(
            movement_samples, dtype=torch.long, device=device
        ),
        "endpoint_3d_target": torch.stack(endpoint_targets),
        "current_position_target": torch.stack(current_position_targets),
        "observed_fraction": torch.tensor(
            observed_fractions, dtype=torch.float32, device=device
        ),
        "target": batch["screen_target"].index_select(0, rows),
        "loss_weight": torch.ones(
            len(chosen), dtype=torch.float32, device=device
        ),
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
    if "session" in batch:
        window["session"] = batch["session"].index_select(0, rows)
    if "canvas" in batch:
        window["canvas_size"] = batch["canvas"].index_select(0, rows)
    elif fallback_canvas is not None:
        window["canvas_size"] = fallback_canvas.to(device).unsqueeze(0).expand(
            len(chosen), -1
        )
    else:
        raise ValueError("canvas size is required for the screen-point loss")
    return window


def _endpoint_losses(
    outputs: dict[str, Any],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    settings = config["model"]["complete_reach"]
    scale = max(float(config["model"].get("trajectory_limit_m", 0.8)), 1e-6)
    beta = float(settings.get("endpoint_huber_beta", 0.05))
    endpoint = F.smooth_l1_loss(
        outputs["endpoint_3d"] / scale,
        window["endpoint_3d_target"] / scale,
        beta=beta,
    )
    consistency = F.smooth_l1_loss(
        outputs["endpoint_3d"] / scale,
        outputs["trajectory"][:, -1] / scale,
        beta=beta,
    )
    return {
        "complete_endpoint_3d": endpoint,
        "complete_endpoint_consistency": consistency,
    }


def student_objective(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    combined = _ROLLING_STUDENT_OBJECTIVE(
        outputs, teacher_outputs, window, config
    )
    settings = config["model"]["complete_reach"]
    fused = _endpoint_losses(outputs, window, config)
    emg = _endpoint_losses(outputs["emg_only"], window, config)
    combined["loss"] = (
        combined["loss"]
        + float(settings.get("endpoint_weight", 0.75))
        * fused["complete_endpoint_3d"]
        + float(settings.get("endpoint_consistency_weight", 0.25))
        * fused["complete_endpoint_consistency"]
        + float(settings.get("emg_endpoint_weight", 0.25))
        * emg["complete_endpoint_3d"]
    )
    combined.update({name: value.detach() for name, value in fused.items()})
    combined.update({
        f"emg_{name}": value.detach() for name, value in emg.items()
    })
    return combined


def _extend(
    totals: dict[str, list[float]], name: str, values: torch.Tensor
) -> None:
    totals.setdefault(name, []).extend(
        values.detach().cpu().reshape(-1).tolist()
    )


@torch.no_grad()
def evaluate_complete_reach(
    model: CompleteReachDistillationModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rate = base.effective_rate(config)
    steps = int(config["model"]["teacher_trajectory_steps"])
    limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    generator = np.random.default_rng(0)
    totals: dict[str, list[float]] = {}
    for batch in loader:
        if batch is None:
            continue
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        for lead in evaluation_leads:
            window = make_complete_reach_window(
                batch,
                context_samples,
                patch_length,
                steps,
                generator,
                limit,
                velocity_scale,
                canvas_tensor,
                fixed_lead=lead,
            )
            if window is None:
                continue
            outputs = model.student_forward(
                window["emg"], window["imu"], window["time_mask"], sample=False
            )
            screen = (
                (outputs["prediction"] - window["target"])
                * window["canvas_size"]
            ).norm(dim=-1)
            endpoint = 100.0 * (
                outputs["endpoint_3d"] - window["endpoint_3d_target"]
            ).norm(dim=-1)
            path = 100.0 * (
                outputs["trajectory"] - window["trajectory_target"]
            ).norm(dim=-1).mean(dim=-1)
            consistency = 100.0 * (
                outputs["endpoint_3d"] - outputs["trajectory"][:, -1]
            ).norm(dim=-1)
            label = int(round(1000.0 * lead / rate))
            for name, values in (
                ("complete_screen_px", screen),
                ("complete_endpoint_3d_cm", endpoint),
                ("complete_path_cm", path),
                ("complete_endpoint_consistency_cm", consistency),
            ):
                _extend(totals, name, values)
                _extend(totals, f"{name}_{label}ms", values)
    return {
        name: float(np.mean(values))
        for name, values in totals.items()
        if values
    }


@torch.no_grad()
def evaluate(
    model: CompleteReachDistillationModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    mean_target: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    metrics = _ROLLING_EVALUATE(
        model,
        loader,
        config,
        context_samples,
        patch_length,
        evaluation_leads,
        canvas_tensor,
        mean_target,
        device,
    )
    metrics.update(evaluate_complete_reach(
        model,
        loader,
        config,
        context_samples,
        patch_length,
        evaluation_leads,
        canvas_tensor,
        device,
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


def main() -> None:
    if not _has_option("--config"):
        sys.argv[1:1] = ["--config", "configs/tracked_complete_reach.yaml"]
    if not _has_option("--output-dir"):
        sys.argv[1:1] = ["--output-dir", "runs/complete_reach"]

    # Install the isolated experiment into the established, tested training
    # loop only for this process.  Old entrypoints retain their exact behavior.
    base.make_distillation_window = make_complete_reach_window
    base.milliseconds_to_samples = milliseconds_to_samples
    base.evaluate = evaluate
    base.train_teacher = bridge.train_teacher
    channel.ChannelHorizonLatentDistillationModel = (
        CompleteReachDistillationModel
    )
    channel.student_objective = student_objective
    channel.__doc__ = __doc__
    print(
        "complete-reach experiment: causal EMG+IMU prefix -> screen endpoint + "
        "onset-relative 3D endpoint + full onset-to-touch path; true 0ms enabled"
    )
    channel.main()

    output = Path(_option_value("--output-dir", "runs/complete_reach"))
    results_path = output / "results.json"
    if not results_path.exists():
        return
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("config", {})
    test = payload.get("test", {})
    rate = base.effective_rate(config)
    leads = tuple(dict.fromkeys(
        milliseconds_to_samples(value, rate)
        for value in config.get("distillation", {}).get(
            "evaluation_leads_ms", [0, 50, 100, 200, 300, 400]
        )
    ))
    print("\n=== complete-reach diagnostics ===")
    print("  prediction convergence as wearable history grows")
    for lead in reversed(leads):
        label = int(round(1000.0 * lead / rate))
        print(
            f"    {label:3d}ms to touch: "
            f"screen={test.get(f'complete_screen_px_{label}ms', float('nan')):7.1f}px "
            f"endpoint={test.get(f'complete_endpoint_3d_cm_{label}ms', float('nan')):6.2f}cm "
            f"full-path={test.get(f'complete_path_cm_{label}ms', float('nan')):6.2f}cm"
        )
    print(
        "  all observations: "
        f"screen={test.get('complete_screen_px', float('nan')):.1f}px | "
        f"endpoint={test.get('complete_endpoint_3d_cm', float('nan')):.2f}cm | "
        f"full-path={test.get('complete_path_cm', float('nan')):.2f}cm | "
        "endpoint/path agreement="
        f"{test.get('complete_endpoint_consistency_cm', float('nan')):.2f}cm"
    )


if __name__ == "__main__":
    main()
