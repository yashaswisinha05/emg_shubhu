#!/usr/bin/env python3
"""Train channel-time EMG attention and a disambiguated horizon latent.

This is an isolated successor to ``train_latent_distillation_model.py`` and
does not use any virtual-leader loss.  It keeps the factor-guided VAE teacher,
shared decoder, EMG-only objective, IMU dropout, and wearable interventions,
then adds:

1. a learnable gate over physical EMG sensors at every causal timestep;
2. a named latent slice supervised to predict time remaining until touch;
3. adversarial removal of horizon information from the other latent slices;
4. final causal per-sensor ablations and channel-by-lookback reports.

VIVE position, velocity, endpoint, and lead time are training/evaluation
labels only.  The deployed ``student_forward`` still accepts only EMG, IMU,
and a causal mask.

Example:

    python scripts/train_channel_horizon_distillation_model.py \
      --root "/media/.../emg_imu_vive" \
      --config configs/tracked_channel_horizon_distillation.yaml \
      --cache-dir artifacts/tracked_cache_posture \
      --session-prefixes dev_a1 dev_a2 dev_a3 dev_a4 \
      --device cuda --teacher-epochs 25 --epochs 50 --finetune-epochs 0 \
      --lead-window-ms 50 400 \
      --output-dir runs/channel_horizon_distillation
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import train_latent_distillation_model as base  # noqa: E402
from emg_touch.data.schema import sensor_names  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    emg_feature_count,
    imu_feature_count,
)
from emg_touch.models.channel_horizon_distillation import (  # noqa: E402
    ChannelHorizonLatentDistillationModel,
    channel_attention_regularizers,
    horizon_guidance_losses,
    physical_sensor_feature_indices,
)


_BASE_TEACHER_OBJECTIVE = base.teacher_objective
_BASE_STUDENT_OBJECTIVE = base.student_objective


def _weighted_horizon(
    losses: dict[str, torch.Tensor], settings: dict[str, Any], prefix: str
) -> torch.Tensor:
    return (
        float(settings.get(f"{prefix}_classification_weight", 0.0))
        * losses["horizon_classification"]
        + float(settings.get(f"{prefix}_regression_weight", 0.0))
        * losses["horizon_regression"]
        + float(settings.get(f"{prefix}_adversarial_weight", 0.0))
        * losses["horizon_adversarial"]
    )


def teacher_objective(
    outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
    kl_weight: float,
) -> dict[str, torch.Tensor]:
    combined = _BASE_TEACHER_OBJECTIVE(outputs, window, config, kl_weight)
    settings = config["model"]["horizon_latent"]
    horizon = horizon_guidance_losses(
        outputs,
        window["lead_samples"],
        base.effective_rate(config),
        settings,
    )
    combined["loss"] = combined["loss"] + _weighted_horizon(
        horizon, settings, "teacher"
    )
    combined.update({name: value.detach() for name, value in horizon.items()})
    return combined


def student_objective(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, torch.Tensor],
    window: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    combined = _BASE_STUDENT_OBJECTIVE(
        outputs, teacher_outputs, window, config
    )
    settings = config["model"]["horizon_latent"]
    rate = base.effective_rate(config)
    fused_horizon = horizon_guidance_losses(
        outputs, window["lead_samples"], rate, settings
    )
    emg_horizon = horizon_guidance_losses(
        outputs["emg_only"], window["lead_samples"], rate, settings
    )
    channel = channel_attention_regularizers(
        outputs["channel_attention"], window["time_mask"]
    )
    channel_settings = config["model"]["channel_time_attention"]
    combined["loss"] = (
        combined["loss"]
        + _weighted_horizon(fused_horizon, settings, "student")
        + _weighted_horizon(emg_horizon, settings, "emg")
        + float(channel_settings.get("entropy_weight", 0.0))
        * channel["channel_entropy"]
        + float(channel_settings.get("smoothness_weight", 0.0))
        * channel["channel_smoothness"]
    )
    combined.update({name: value.detach() for name, value in fused_horizon.items()})
    combined.update({
        f"emg_{name}": value.detach() for name, value in emg_horizon.items()
    })
    combined.update({name: value.detach() for name, value in channel.items()})
    return combined


def _extend(
    store: dict[str, list[float]], name: str, values: torch.Tensor
) -> None:
    store.setdefault(name, []).extend(values.detach().cpu().reshape(-1).tolist())


@torch.no_grad()
def evaluate_disambiguation(
    model: ChannelHorizonLatentDistillationModel,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    context_samples: int,
    patch_length: int,
    evaluation_leads: tuple[int, ...],
    canvas_tensor: torch.Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    """Measure learned gates, time-to-go, and causal sensor contributions."""
    model.eval()
    rate = base.effective_rate(config)
    teacher_steps = int(config["model"]["teacher_trajectory_steps"])
    trajectory_limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    sensors = sensor_names(config["data"])
    emg_channels = emg_feature_count(config["data"])
    lag_edges = list(
        map(
            float,
            config["model"]["channel_time_attention"].get(
                "report_lag_edges_ms", [0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0]
            ),
        )
    )
    if len(lag_edges) < 2:
        raise ValueError("report_lag_edges_ms needs at least two edges")
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
            window = base.make_distillation_window(
                batch,
                context_samples,
                patch_length,
                teacher_steps,
                generator,
                trajectory_limit,
                velocity_scale,
                canvas_tensor,
                fixed_lead=lead,
            )
            if window is None:
                continue
            output = model.student_forward(
                window["emg"],
                window["imu"],
                window["time_mask"],
                sample=False,
                include_emg_only=True,
            )
            guidance = output["guidance"]
            predicted_ms = guidance["horizon_expected_ms"]
            actual_ms = (
                1000.0 * window["lead_samples"].to(torch.float32) / rate
            )
            _extend(totals, "horizon_absolute_error_ms", (predicted_ms - actual_ms).abs())
            lead_label = int(round(1000.0 * lead / rate))
            _extend(totals, f"horizon_prediction_at_{lead_label}ms", predicted_ms)

            centers = model.guidance.horizon.centers_ms.to(predicted_ms)
            predicted_bin = guidance["horizon_logits"].argmax(dim=-1)
            actual_bin = (actual_ms.unsqueeze(-1) - centers).abs().argmin(dim=-1)
            _extend(
                totals,
                "horizon_bin_accuracy",
                (predicted_bin == actual_bin).to(torch.float32),
            )

            attention = output["channel_attention"]
            valid = window["time_mask"]
            for sensor_index, sensor in enumerate(sensors):
                _extend(
                    totals,
                    f"channel_attention_{sensor}",
                    attention[:, :, sensor_index][valid],
                )

            steps = attention.size(1)
            lag_ms = (
                torch.arange(steps - 1, -1, -1, device=device, dtype=torch.float32)
                * 1000.0
                / rate
            )
            for low, high in zip(lag_edges[:-1], lag_edges[1:]):
                in_lag = (lag_ms >= low) & (lag_ms < high)
                bin_mask = valid & in_lag.unsqueeze(0)
                label = f"{int(low)}_{int(high)}ms"
                if bin_mask.any():
                    for sensor_index, sensor in enumerate(sensors):
                        _extend(
                            totals,
                            f"channel_attention_{sensor}_{label}",
                            attention[:, :, sensor_index][bin_mask],
                        )

            for sensor_index, sensor in enumerate(sensors):
                ablated_emg = window["emg"].clone()
                indices = physical_sensor_feature_indices(
                    emg_channels, len(sensors), sensor_index
                )
                ablated_emg[:, :, indices] = 0.0
                ablated = model.student_forward(
                    ablated_emg,
                    window["imu"],
                    window["time_mask"],
                    sample=False,
                )
                pixel = (
                    (ablated["prediction"] - window["target"])
                    * window["canvas_size"]
                ).norm(dim=-1)
                _extend(totals, f"without_sensor_{sensor}_px", pixel)

    return {
        name: float(np.mean(values))
        for name, values in totals.items()
        if values
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--config", default="configs/tracked_channel_horizon_distillation.yaml"
    )
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache_posture")
    parser.add_argument("--output-dir", default="runs/channel_horizon_distillation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--teacher-epochs", type=int)
    parser.add_argument("--epochs", type=int, help="Student distillation epochs")
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument(
        "--lead-window-ms", type=float, nargs=2, metavar=("MIN", "MAX")
    )
    parser.add_argument("--session-prefixes", nargs="+")
    parser.add_argument("--no-adaptive-sampling", action="store_true")
    args = parser.parse_args()

    config = base.load_config(args.config)
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
    base.seed_everything(seed)
    device = base.choose_device(args.device)
    train_loader, validation_loader, test_loader = base.build_experiment_loaders(
        config, args.root, Path(args.cache_dir)
    )
    rate = base.effective_rate(config)
    model_config = config["model"]
    context_samples = max(
        int(model_config["patch_length"]),
        base.milliseconds_to_samples(
            float(model_config.get("context_ms", 2000.0)), rate
        ),
    )
    low_ms, high_ms = sorted(
        map(float, config["distillation"].get("lead_window_ms", [50.0, 400.0]))
    )
    lead_window = (
        base.milliseconds_to_samples(low_ms, rate),
        base.milliseconds_to_samples(high_ms, rate),
    )
    evaluation_leads = tuple(dict.fromkeys(
        base.milliseconds_to_samples(value, rate)
        for value in config["distillation"].get(
            "evaluation_leads_ms", [50.0, 100.0, 200.0, 300.0, 400.0]
        )
    ))
    selection_lead = base.milliseconds_to_samples(
        float(config["distillation"].get("selection_lead_ms", 200.0)), rate
    )
    fallback = base.canvas_from_disk(args.root)
    canvas_tensor = (
        torch.tensor(fallback, dtype=torch.float32, device=device)
        if fallback else None
    )
    mean_target = base.training_mean_target(train_loader)
    model = ChannelHorizonLatentDistillationModel(
        config,
        emg_feature_count(config["data"]),
        imu_feature_count(config["data"]),
    ).to(device)

    parameters = set(inspect.signature(model.student_forward).parameters)
    forbidden = {"position", "velocity", "trajectory_features", "lead_samples"}
    assert not (parameters & forbidden)
    print(
        "deployment input check: channel gates and horizon are inferred from "
        "EMG+IMU history; VIVE and true lead are labels only"
    )
    print(
        f"train {len(train_loader.dataset)} | val {len(validation_loader.dataset)} "
        f"| test {len(test_loader.dataset)} trials"
    )
    print(
        f"context {context_samples} samples ({context_samples / rate:.2f}s), "
        f"lead {low_ms:.0f}-{high_ms:.0f}ms, latent {model_config['latent_dim']}D"
    )
    horizon = model.guidance.horizon
    print(
        f"horizon latent z[{horizon.start}:{horizon.end}] with bins "
        f"{horizon.centers_ms.cpu().tolist()} ms; physical EMG sensors "
        f"{list(sensor_names(config['data']))}"
    )
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adaptive_settings = config["distillation"].get("adaptive_sampling", {})
    adaptive = (
        bool(adaptive_settings.get("enabled", True))
        and not args.no_adaptive_sampling
    )
    difficulty = base.AdaptiveTrialDifficulty(
        alpha=float(adaptive_settings.get("ema_alpha", 0.5)),
        uniform_mix=float(adaptive_settings.get("uniform_mix", 0.7)),
        power=float(adaptive_settings.get("power", 1.0)),
        max_ratio=float(adaptive_settings.get("max_ratio", 4.0)),
    )

    # Install the additional objectives only in this process. Running the
    # original trainer continues to reproduce the 228.6 px baseline.
    base.teacher_objective = teacher_objective
    base.student_objective = student_objective
    history = base.train_teacher(
        model,
        train_loader,
        validation_loader,
        config,
        int(config["distillation"].get("teacher_epochs", 25)),
        context_samples,
        int(model_config["patch_length"]),
        lead_window,
        selection_lead,
        canvas_tensor,
        device,
        output,
        difficulty,
        adaptive,
    )
    student_history, _, _ = base.train_student_phase(
        "student",
        model,
        train_loader,
        validation_loader,
        config,
        int(config["training"]["epochs"]),
        context_samples,
        int(model_config["patch_length"]),
        lead_window,
        evaluation_leads,
        canvas_tensor,
        mean_target,
        device,
        output,
        difficulty,
        adaptive,
        unfreeze_decoder=False,
    )
    history.extend(student_history)

    finetune_epochs = int(config["distillation"].get("finetune_epochs", 0))
    if finetune_epochs > 0:
        finetune_history, _, _ = base.train_student_phase(
            "finetune",
            model,
            train_loader,
            validation_loader,
            config,
            finetune_epochs,
            context_samples,
            int(model_config["patch_length"]),
            lead_window,
            evaluation_leads,
            canvas_tensor,
            mean_target,
            device,
            output,
            difficulty,
            adaptive,
            unfreeze_decoder=True,
        )
        history.extend(finetune_history)

    test = base.evaluate(
        model,
        test_loader,
        config,
        context_samples,
        int(model_config["patch_length"]),
        evaluation_leads,
        canvas_tensor,
        mean_target,
        device,
    )
    test.update(evaluate_disambiguation(
        model,
        test_loader,
        config,
        context_samples,
        int(model_config["patch_length"]),
        evaluation_leads,
        canvas_tensor,
        device,
    ))

    print("\n=== channel + horizon wearable-only test ===")
    for name in (
        "student", "emg_only", "without_emg", "shuffled_emg", "without_imu",
        "mean", "teacher",
    ):
        key = f"{name}_px"
        if key in test:
            suffix = " (training-only oracle)" if name == "teacher" else ""
            print(f"  {name:13}: {test[key]:7.1f} px{suffix}")

    full = test.get("student_px", float("nan"))
    print("\n  paired modality contribution (positive means it helps)")
    for label, key in (
        ("remove EMG", "without_emg_px"),
        ("shuffle EMG", "shuffled_emg_px"),
        ("remove IMU", "without_imu_px"),
    ):
        if key in test:
            print(f"    {label:11}: {test[key] - full:+7.1f} px")

    print("\n  student endpoint error by true lead")
    for lead in evaluation_leads:
        key = f"student_lead_{lead}_px"
        if key in test:
            print(f"    {1000.0 * lead / rate:5.0f} ms: {test[key]:7.1f} px")

    print("\n  causal physical-sensor contribution (positive means it helps)")
    for sensor in sensor_names(config["data"]):
        without = test.get(f"without_sensor_{sensor}_px", float("nan"))
        print(f"    {sensor:4}: {without - full:+7.1f} px (without={without:.1f}px)")

    print(
        f"\n  predicted time-to-touch MAE: "
        f"{test.get('horizon_absolute_error_ms', float('nan')):.1f} ms | "
        f"bin accuracy: {100.0 * test.get('horizon_bin_accuracy', float('nan')):.1f}%"
    )
    for lead in evaluation_leads:
        label = int(round(1000.0 * lead / rate))
        key = f"horizon_prediction_at_{label}ms"
        if key in test:
            print(f"    true {label:3} ms -> predicted {test[key]:6.1f} ms")

    print("\n  learned channel attention by lookback (share across sensors)")
    lag_edges = config["model"]["channel_time_attention"].get(
        "report_lag_edges_ms", [0, 100, 250, 500, 1000, 2000]
    )
    for low, high in zip(lag_edges[:-1], lag_edges[1:]):
        label = f"{int(low)}_{int(high)}ms"
        shares = []
        for sensor in sensor_names(config["data"]):
            value = test.get(f"channel_attention_{sensor}_{label}", float("nan"))
            shares.append(f"{sensor}={100.0 * value:5.1f}%")
        print(f"    {int(low):4}-{int(high):4} ms: " + "  ".join(shares))

    if "intent_target_accuracy" in test:
        print("\n  factor-guided latent probes")
        print(
            f"    fused intent -> target : "
            f"{100.0 * test['intent_target_accuracy']:6.2f}%"
        )
        print(
            f"    EMG intent -> target   : "
            f"{100.0 * test['emg_intent_target_accuracy']:6.2f}%"
        )
        print(
            f"    other -> target leak   : "
            f"{100.0 * test['other_target_leakage_accuracy']:6.2f}%"
        )
    print(
        f"\n  student trajectory: "
        f"{test.get('student_trajectory_cm', float('nan')):.2f} cm | "
        f"IMU-only tracking: "
        f"{test.get('imu_tracking_trajectory_cm', float('nan')):.2f} cm | "
        f"latent RMSE: {test.get('latent_rmse', float('nan')):.4f}"
    )

    base.save_json(
        {"config": config, "history": history, "test": test},
        output / "results.json",
    )
    torch.save(
        {"model_state": base.clone_state(model), "config": config, "test": test},
        output / "final.pt",
    )
    print(f"wrote {output / 'results.json'} and {output / 'final.pt'}")


if __name__ == "__main__":
    main()
