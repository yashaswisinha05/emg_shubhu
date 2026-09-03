#!/usr/bin/env python3
"""Fine-tune a trained channel/horizon distillation checkpoint to lean on EMG more.

Does NOT modify train_latent_distillation_model.py, train_channel_horizon_
distillation_model.py, or either model file. Follows the exact pattern the
channel/horizon script itself used to extend the base script: monkey-patch
`base.student_objective`, then drive training through the existing, tested
`base.train_student_phase("finetune", ...)` - no training loop reimplemented.

    python scripts/finetune_emg_necessity.py \\
        --root "/media/.../emg_imu_vive" \\
        --checkpoint runs/channel_horizon_distillation/final.pt \\
        --cache-dir artifacts/tracked_cache_posture \\
        --device cuda --epochs 20 --necessity-weight 1.0 --margin 0.3 \\
        --output-dir runs/emg_necessity_finetune

WHAT THIS ADDS, and why it targets exactly what was asked ("make EMG take
EMG input more"). The evaluation this pipeline already runs computes and
prints `without_emg` (EMG zeroed) against `student` (real EMG) as a
diagnostic - "remove EMG: +25.5px" in the run this is built from. That
number was never fed back into training; it only ever got measured after
the fact. This adds a loss term that trains it directly: every step, run
the SAME student a second time with EMG zeroed (torch.no_grad, then
detached again defensively - a stop-gradient reference, the same pattern
this project's vae_discriminator.py IMU-only critic already established:
beat a reference without being able to cheat by degrading it), then a
hinge loss requires the real-EMG prediction to beat it by --margin:

    relu(margin + real_error - ablated_error.detach())

in the SAME pixel_normalizer_px-normalised units grid_point_loss's own
pixel/radial terms already use (divide by config["loss"]["pixel_normalizer_
px"], default 80 px) - checked against the existing loss composition before
picking a weight, not assumed: mixing raw-pixel-scale and normalised-scale
terms in one sum would make necessity_weight's meaning depend on an
implicit, undocumented unit conversion. Gradient flows only through the
real-EMG path (ablated_error is detached), so the loss can only be reduced
by making EMG-equipped predictions better - never by making the reference
worse. Training and evaluation now measure literally the same quantity.

necessity_weight ramps from 0 over the first third of TOTAL TRAINING STEPS,
implemented as a self-contained step counter inside the objective closure
rather than by calling train_student_phase repeatedly for one epoch at a
time. That second approach was tried first and rejected after reading
train_student_phase itself: its optimizer, its ReduceLROnPlateau scheduler,
and its early-stopping "stale" counter are all constructed INSIDE the
function and reset on every call - calling it once per epoch would discard
Adam's momentum and restart the LR schedule's patience count on every
single epoch. Calling it exactly once for the full --epochs, with the ramp
computed internally, is what fixes that.

Also turns on channel_time_attention.entropy_weight (0.0 in the existing
config, confirmed by reading it - not a value that had simply not been
tuned yet, an unweighted term contributing nothing). The physical-sensor
gate already exists and starts as an exact identity by design; with the
weight at 0, nothing pushes it away from that start. The existing causal
ablation shows S8 net-HARMFUL (-2.5px: removing it helps) while the printed
channel attention sits at a flat 23-27% across all four sensors - consistent
with a gate that has had no incentive to depart from where it started. This
does not hand-pick which sensor should matter; it lets the already-built
selectivity regulariser actually push.

The teacher is frozen for this stage - train_student_phase's own convention
for "finetune". This changes how the student uses EMG, not what the
teacher (a training-only oracle) target represents.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import train_latent_distillation_model as base  # noqa: E402
from scripts import train_channel_horizon_distillation_model as chd  # noqa: E402
from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    emg_feature_count,
    imu_feature_count,
)
from emg_touch.models.channel_horizon_distillation import (  # noqa: E402
    ChannelHorizonLatentDistillationModel,
)
from emg_touch.utils import choose_device, save_json, seed_everything  # noqa: E402


class NecessityRamp:
    """Self-incrementing step counter - lives inside the objective closure so
    a single train_student_phase call sees a smooth ramp with no external
    per-epoch bookkeeping (see module docstring for why an external loop
    calling train_student_phase repeatedly was rejected: it resets that
    function's own optimizer/scheduler/early-stopping state every call).
    """

    def __init__(self, target: float, ramp_steps: int) -> None:
        self.target = target
        self.ramp_steps = max(1, ramp_steps)
        self.step = 0

    def value(self) -> float:
        self.step += 1
        return self.target * min(1.0, self.step / self.ramp_steps)


def make_emg_necessity_objective(model, ramp: NecessityRamp, margin: float, pixel_unit: float):
    """Wraps the ALREADY-INSTALLED student_objective (chd's channel/horizon
    version - itself already a wrapper around the base script's own), adding
    the necessity hinge on top.
    """
    base_objective = chd.student_objective

    def wrapped(outputs, teacher_outputs, window, config):
        combined = base_objective(outputs, teacher_outputs, window, config)
        with torch.no_grad():
            ablated = model.student_forward(
                torch.zeros_like(window["emg"]), window["imu"], window["time_mask"],
                sample=False,
            )
        canvas = window["canvas_size"]
        real_error_px = ((outputs["prediction"] - window["target"]) * canvas).norm(dim=-1)
        ablated_error_px = (
            (ablated["prediction"].detach() - window["target"]) * canvas
        ).norm(dim=-1)
        real_scaled = real_error_px / pixel_unit
        ablated_scaled = ablated_error_px.detach() / pixel_unit
        weight = ramp.value()
        necessity = torch.relu(margin + real_scaled - ablated_scaled).mean()
        combined["loss"] = combined["loss"] + weight * necessity
        combined["emg_necessity"] = necessity.detach()
        combined["emg_gap_px"] = (ablated_error_px - real_error_px).detach().mean()
        return combined

    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Override the config stored in the checkpoint.")
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache_posture")
    parser.add_argument("--output-dir", default="runs/emg_necessity_finetune")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--necessity-weight", type=float, default=1.0,
                        help="Target value the ramp reaches at 1/3 of total steps. "
                        "Same O(1) scale as this pipeline's own emg_only_weight/"
                        "latent_distillation_weight, since the loss itself is now "
                        "normalised by pixel_normalizer_px to match.")
    parser.add_argument("--margin", type=float, default=0.3,
                        help="In pixel_normalizer_px units (default 80px -> 0.3 "
                        "means ~24 px), matching how grid_point_loss's own pixel "
                        "terms are scaled. The currently measured gap is ~25.5 px "
                        "= 0.32 in these units, so this starts close to already-met.")
    parser.add_argument("--channel-entropy-weight", type=float, default=0.05,
                        help="0.0 in the existing config - turns the already-built "
                        "selectivity regulariser on.")
    parser.add_argument("--channel-smoothness-weight", type=float, default=0.01)
    parser.add_argument("--no-adaptive-sampling", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_config(args.config) if args.config else checkpoint["config"]
    config["distillation"] = dict(config.get("distillation", {}))
    config["model"] = dict(config["model"])
    config["model"]["channel_time_attention"] = dict(
        config["model"].get("channel_time_attention", {})
    )
    config["model"]["channel_time_attention"]["entropy_weight"] = args.channel_entropy_weight
    config["model"]["channel_time_attention"]["smoothness_weight"] = args.channel_smoothness_weight
    pixel_unit = float(config.get("loss", {}).get("pixel_normalizer_px", 80.0))

    seed_everything(int(args.seed if args.seed is not None else config.get("seed", 42)))
    device = choose_device(args.device)

    train_loader, validation_loader, test_loader = base.build_experiment_loaders(
        config, args.root, Path(args.cache_dir)
    )
    rate = base.effective_rate(config)
    model_config = config["model"]
    context_samples = max(
        int(model_config["patch_length"]),
        base.milliseconds_to_samples(float(model_config.get("context_ms", 2000.0)), rate),
    )
    lead_ms = config["distillation"].get("lead_window_ms", [50.0, 400.0])
    low_ms, high_ms = sorted(map(float, lead_ms))
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
    fallback = base.canvas_from_disk(args.root)
    canvas_tensor = (
        torch.tensor(fallback, dtype=torch.float32, device=device) if fallback else None
    )
    mean_target = base.training_mean_target(train_loader)

    model = ChannelHorizonLatentDistillationModel(
        config, emg_feature_count(config["data"]), imu_feature_count(config["data"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"loaded {args.checkpoint}")
    if "test" in checkpoint and "student_px" in checkpoint["test"]:
        print(f"checkpoint's own reported test: student={checkpoint['test']['student_px']:.1f}px")

    ramp_steps = max(1, len(train_loader) * args.epochs // 3)
    ramp = NecessityRamp(args.necessity_weight, ramp_steps)
    base.student_objective = make_emg_necessity_objective(model, ramp, args.margin, pixel_unit)
    print(f"student_objective monkey-patched: base -> chd (channel/horizon) -> "
          f"+ EMG-necessity hinge (margin={args.margin} [{args.margin * pixel_unit:.0f}px], "
          f"target weight={args.necessity_weight}, ramped over {ramp_steps} steps)")
    print(f"channel_time_attention entropy_weight={args.channel_entropy_weight} "
          f"smoothness_weight={args.channel_smoothness_weight} (was 0.0/unset)")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adaptive_settings = config["distillation"].get("adaptive_sampling", {})
    adaptive = (
        bool(adaptive_settings.get("enabled", True)) and not args.no_adaptive_sampling
    )
    difficulty = base.AdaptiveTrialDifficulty(
        alpha=float(adaptive_settings.get("ema_alpha", 0.5)),
        uniform_mix=float(adaptive_settings.get("uniform_mix", 0.7)),
        power=float(adaptive_settings.get("power", 1.0)),
        max_ratio=float(adaptive_settings.get("max_ratio", 4.0)),
    )

    # Exactly one call, exactly like the phases in the pipeline this builds
    # on - see module docstring for why per-epoch calls were rejected.
    history, _, _ = base.train_student_phase(
        "emg_necessity", model, train_loader, validation_loader, config,
        args.epochs, context_samples, int(model_config["patch_length"]), lead_window,
        evaluation_leads, canvas_tensor, mean_target, device, output,
        difficulty, adaptive, unfreeze_decoder=True,
    )

    test = base.evaluate(
        model, test_loader, config, context_samples, int(model_config["patch_length"]),
        evaluation_leads, canvas_tensor, mean_target, device,
    )
    print("\n=== after EMG-necessity fine-tuning ===")
    for name in ("student", "emg_only", "without_emg", "shuffled_emg", "without_imu", "mean"):
        if f"{name}_px" in test:
            print(f"  {name:12}: {test[f'{name}_px']:7.1f} px")
    if "student_px" in test and "without_emg_px" in test:
        gap = test["without_emg_px"] - test["student_px"]
        print(f"\n  remove EMG gap: {gap:+.1f} px (compare to +25.5 px before this run)")

    save_json({"config": config, "history": history, "test": test}, output / "results.json")
    torch.save({"model_state": model.state_dict(), "config": config, "test": test},
              output / "final.pt")
    print(f"\nwrote {output / 'results.json'} and {output / 'final.pt'} "
          f"(original checkpoint at {args.checkpoint} untouched)")


if __name__ == "__main__":
    main()
