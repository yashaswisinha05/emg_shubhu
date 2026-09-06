#!/usr/bin/env python3
"""The pixel-touch counterpart: let EMG+IMU choose how early to commit.

train_adaptive_horizon_model.py let the network choose how far AHEAD to
forecast trajectory. This is the mirror image for screen-touch prediction:
tau = how many ms BEFORE TOUCH the network trusts its own prediction right
now, chosen per trial from EMG+IMU rather than fixed by --lead-window-ms.

    python scripts/train_adaptive_lead_model.py \\
        --root "/media/.../emg_imu_vive" \\
        --config configs/tracked_grid_vae_discriminator.yaml \\
        --cache-dir artifacts/tracked_cache_posture \\
        --device cuda --epochs 20 \\
        --tau-min-ms 50 --tau-max-ms 1000 \\
        --output-dir runs/adaptive_lead

Why this needed genuinely different machinery from the trajectory version,
not just a relabelling: the trajectory model emits one rollout per forward
pass and interpolate_at reads two points out of it - the SAME computation
answers every possible tau. The pointing model has no such rollout: how
early to commit changes what the ENCODER SEES (how much causal history),
not just what is read out of a fixed computation. So there is nothing to
gather from - this runs the encoder TWICE per training step, once at
floor(tau) and once at ceil(tau) samples of lead time, and blend_two
(adaptive_horizon.py) linearly interpolates the two resulting PREDICTIONS
by tau's fractional part. Same principle as interpolate_at, no gather.

Reuses EMGImportanceVAE (vae_discriminator.py) as the underlying predictor,
unmodified - the tau head reads mu_z, the VAE's own latent mean, which is
already a complete summary of a window's EMG+IMU content computed for
another purpose, rather than adding yet another representation to learn.
The IMU-only critic inside EMGImportanceVAE keeps training exactly as
before; this script adds an ADDITIONAL loss term on top of the model's own,
same pattern as every other tour through this codebase.

Same ramp, same reasoning, same failure this project already found once for
the trajectory case: reach_weight ramps from 0 over the first third of
training, because an un-ramped run risks tau saturating to one constant
value for every input before the accuracy-dependent gradient (which only
differentiates once tau is already in a sensitive range) gets a chance to
shape anything - see adaptive_horizon.py's module docstring for the
synthetic control that demonstrated exactly this collapse and the ramp
that fixed it.

Reports the same decisive comparison: error at the network's own tau
against error at a FIXED tau_min and a FIXED tau_max for every trial. If
adaptive does not beat both, per-example choice bought nothing here either.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    CANVAS_COLUMNS,
    build_tracked_loaders,
    emg_feature_count,
    imu_feature_count,
)
from emg_touch.grid_training import grid_point_loss  # noqa: E402
from emg_touch.models.adaptive_horizon import AdaptiveHorizonHead, blend_two  # noqa: E402
from emg_touch.models.disentangle import reversal_strength  # noqa: E402
from emg_touch.models.vae_discriminator import EMGImportanceVAE  # noqa: E402
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
            .apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
        )
        if len(values):
            return float(values[-1][0]), float(values[-1][1])
    return None


def window_at_lead(
    batch: dict, lead_samples: torch.Tensor, minimum_prefix: int, patch_length: int,
    fallback_canvas: torch.Tensor | None,
) -> dict:
    """Per-row window cut at touch - lead_samples[row], clamped into range.

    lead_samples is a per-row LongTensor - every row can want a different
    amount of history, which is the entire point of a per-example tau.
    Clamped (max(start, min(cut, latest))) rather than dropped, matching
    make_grid_window's own convention: a row that cannot supply the exact
    requested lead still contributes at whatever lead it CAN supply, rather
    than vanishing from the batch and silently shrinking it - the class of
    bug already found and fixed in the trajectory horizon path.
    """
    lengths, onsets = batch["lengths"], batch["onset"]
    device = batch["position"].device
    cuts = []
    for row in range(len(lengths)):
        length = int(lengths[row])
        touch = length - 1
        start = max(int(onsets[row]) + minimum_prefix, minimum_prefix)
        latest = touch - minimum_prefix
        cut = touch - int(lead_samples[row])
        cuts.append(max(start, min(cut, max(start, latest))))

    prefix_length = max(patch_length, min(minimum_prefix * 4, min(cuts)))
    window: dict[str, torch.Tensor] = {}
    for key in ("emg", "imu"):
        window[key] = torch.stack(
            [batch[key][row, cut - prefix_length : cut] for row, cut in enumerate(cuts)]
        )
    window["time_mask"] = torch.ones(len(cuts), prefix_length, dtype=torch.bool, device=device)
    window["target"] = batch["screen_target"]
    if "canvas" in batch:
        window["canvas_size"] = batch["canvas"]
    elif fallback_canvas is not None:
        window["canvas_size"] = fallback_canvas.to(device).unsqueeze(0).expand(len(cuts), -1)
    return window


@torch.no_grad()
def evaluate(model, head, loader, tau_min, tau_max, minimum_prefix, patch_length,
             canvas_tensor, device) -> dict:
    model.eval()
    head.eval()
    adaptive_px, fixed_min_px, fixed_max_px, tau_values = [], [], [], []
    for batch in loader:
        if batch is None:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        B = batch["position"].size(0)
        reference = window_at_lead(
            batch, torch.full((B,), tau_max, dtype=torch.long), minimum_prefix,
            patch_length, canvas_tensor,
        )
        mu_z, _, _ = model.encode(reference["emg"], reference["imu"], reference["time_mask"])
        tau = head(mu_z)
        floor_lead = tau.floor().long().clamp(int(tau_min), int(tau_max) - 1)
        ceil_lead = (floor_lead + 1).clamp(int(tau_min), int(tau_max))

        window_floor = window_at_lead(batch, floor_lead, minimum_prefix, patch_length, canvas_tensor)
        window_ceil = window_at_lead(batch, ceil_lead, minimum_prefix, patch_length, canvas_tensor)
        prediction_floor = model(window_floor["emg"], window_floor["imu"], window_floor["time_mask"])["prediction"]
        prediction_ceil = model(window_ceil["emg"], window_ceil["imu"], window_ceil["time_mask"])["prediction"]
        blended = blend_two(prediction_floor, prediction_ceil, tau, floor_lead.float())
        canvas = reference.get("canvas_size")
        target = reference["target"]

        adaptive_px.extend(((blended - target) * canvas).norm(dim=-1).cpu().tolist())
        window_min = window_at_lead(batch, torch.full((B,), int(tau_min), dtype=torch.long), minimum_prefix, patch_length, canvas_tensor)
        window_max = window_at_lead(batch, torch.full((B,), int(tau_max) - 1, dtype=torch.long), minimum_prefix, patch_length, canvas_tensor)
        pred_min = model(window_min["emg"], window_min["imu"], window_min["time_mask"])["prediction"]
        pred_max = model(window_max["emg"], window_max["imu"], window_max["time_mask"])["prediction"]
        fixed_min_px.extend(((pred_min - target) * canvas).norm(dim=-1).cpu().tolist())
        fixed_max_px.extend(((pred_max - target) * canvas).norm(dim=-1).cpu().tolist())
        tau_values.append(tau.cpu().numpy())

    if not tau_values:
        return {}
    tau_all = np.concatenate(tau_values)
    return {
        "adaptive_px": float(np.mean(adaptive_px)),
        "fixed_tau_min_px": float(np.mean(fixed_min_px)),
        "fixed_tau_max_px": float(np.mean(fixed_max_px)),
        "tau_mean_ms": float(tau_all.mean()), "tau_std_ms": float(tau_all.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/tracked_grid_vae_discriminator.yaml")
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache")
    parser.add_argument("--output-dir", default="runs/adaptive_lead")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tau-min-ms", type=float, default=50.0)
    parser.add_argument("--tau-max-ms", type=float, default=1000.0)
    parser.add_argument("--reach-weight", type=float, default=0.05)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    seed_everything(int(args.seed if args.seed is not None else config.get("seed", 42)))
    device = choose_device(args.device)

    train_loader, validation_loader, test_loader = build_tracked_loaders(
        config, args.root, Path(args.cache_dir)
    )
    print(f"train {len(train_loader.dataset)} | val {len(validation_loader.dataset)} "
          f"| test {len(test_loader.dataset)} trials")

    rate = float(config["data"]["sample_rate_hz"]) / max(1, int(config["data"].get("decimation", 10)))
    minimum_prefix = int(config["virtual_leader"]["minimum_prefix"])
    patch_length = int(config["model"]["patch_length"])
    tau_min = max(1, int(round(args.tau_min_ms * rate / 1000.0)))
    tau_max = max(tau_min + 2, int(round(args.tau_max_ms * rate / 1000.0)))
    print(f"tau range: {args.tau_min_ms:.0f}-{args.tau_max_ms:.0f} ms "
          f"({tau_min}-{tau_max} samples at {rate:.1f} Hz)")

    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = EMGImportanceVAE(config, emg_channels, imu_channels).to(device)
    head = AdaptiveHorizonHead(model.latent_dim, tau_min, tau_max).to(device)

    fallback = canvas_from_disk(args.root)
    if fallback:
        print(f"canvas {fallback[0]:.0f} x {fallback[1]:.0f} px")
    canvas_tensor = (
        torch.tensor(fallback, dtype=torch.float32, device=device) if fallback else None
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    ramp_steps = max(1, len(train_loader) * int(config["training"]["epochs"]) // 3)
    global_step = 0
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best, best_state, history = float("inf"), None, []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        head.train()
        running = {"main": [], "accuracy_at_tau": [], "reach_penalty": []}
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            if batch is None:
                continue
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            B = batch["position"].size(0)
            reach_weight = args.reach_weight * min(1.0, global_step / ramp_steps)

            reference = window_at_lead(
                batch, torch.full((B,), tau_max, dtype=torch.long), minimum_prefix,
                patch_length, canvas_tensor,
            )
            outputs_reference = model(reference["emg"], reference["imu"], reference["time_mask"])
            main_losses = grid_point_loss(outputs_reference, reference, config)
            critic_losses = grid_point_loss(outputs_reference["critic"], reference, config)

            mu_z, _, _ = model.encode(reference["emg"], reference["imu"], reference["time_mask"])
            tau = head(mu_z)
            floor_lead = tau.detach().floor().long().clamp(tau_min, tau_max - 1)
            ceil_lead = (floor_lead + 1).clamp(tau_min, tau_max)

            window_floor = window_at_lead(batch, floor_lead, minimum_prefix, patch_length, canvas_tensor)
            window_ceil = window_at_lead(batch, ceil_lead, minimum_prefix, patch_length, canvas_tensor)
            prediction_floor = model(window_floor["emg"], window_floor["imu"], window_floor["time_mask"])["prediction"]
            prediction_ceil = model(window_ceil["emg"], window_ceil["imu"], window_ceil["time_mask"])["prediction"]
            blended = blend_two(prediction_floor, prediction_ceil, tau, floor_lead.float())

            target = reference["target"]
            accuracy_at_tau = (blended - target).norm(dim=-1).mean()
            reach_penalty = (1.0 - tau / float(tau_max)).mean()
            adaptive_total = accuracy_at_tau + reach_weight * reach_penalty

            total = main_losses["loss"] + critic_losses["loss"] + adaptive_total

            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()),
                float(config["training"].get("gradient_clip_norm", 1.0)),
            )
            optimizer.step()
            running["main"].append(float(main_losses["loss"]))
            running["accuracy_at_tau"].append(float(accuracy_at_tau))
            running["reach_penalty"].append(float(reach_penalty))

        scores = evaluate(model, head, validation_loader, tau_min, tau_max, minimum_prefix,
                          patch_length, canvas_tensor, device)
        history.append({"epoch": epoch, **{f"train_{k}": float(np.mean(v or [0]))
                                            for k, v in running.items()}, **scores})
        print(
            f"epoch={epoch} main={np.mean(running['main'] or [0]):.3f} reach_w={reach_weight:.4f} | "
            f"val adaptive={scores.get('adaptive_px', float('nan')):.1f}px "
            f"fixed@min={scores.get('fixed_tau_min_px', float('nan')):.1f}px "
            f"fixed@max={scores.get('fixed_tau_max_px', float('nan')):.1f}px | "
            f"tau={scores.get('tau_mean_ms', float('nan')):.0f}±{scores.get('tau_std_ms', float('nan')):.0f}ms"
        )
        if scores.get("adaptive_px", float("inf")) < best:
            best = scores["adaptive_px"]
            best_state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
            torch.save({**best_state, "config": config, "tau_min": tau_min, "tau_max": tau_max},
                      output / "best.pt")

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        head.load_state_dict(best_state["head_state"])
    test = evaluate(model, head, test_loader, tau_min, tau_max, minimum_prefix, patch_length,
                    canvas_tensor, device)

    print("\n=== test ===")
    print(f"  adaptive (network's own tau) : {test.get('adaptive_px', float('nan')):6.1f} px")
    print(f"  fixed at tau_min ({args.tau_min_ms:.0f} ms)     : {test.get('fixed_tau_min_px', float('nan')):6.1f} px")
    print(f"  fixed at tau_max ({args.tau_max_ms:.0f} ms)     : {test.get('fixed_tau_max_px', float('nan')):6.1f} px")
    print(f"  tau chosen: {test.get('tau_mean_ms', float('nan')):.0f} ± {test.get('tau_std_ms', float('nan')):.0f} ms")

    if test.get("tau_std_ms", 0) < 0.02 * (args.tau_max_ms - args.tau_min_ms):
        print("\n  tau_std is small relative to the tau range - may have collapsed to a")
        print("  near-constant regardless of input. Try adjusting --reach-weight.")
    beats_both = (
        test.get("adaptive_px", 1e9) < test.get("fixed_tau_min_px", 0)
        and test.get("adaptive_px", 1e9) < test.get("fixed_tau_max_px", 0)
    )
    print(f"\n  adaptive beats BOTH fixed extremes: "
          f"{'yes' if beats_both else 'no - the per-example choice is not earning its complexity here'}")

    save_json({"history": history, "test": test}, output / "results.json")
    print(f"\nwrote {output / 'results.json'}")


if __name__ == "__main__":
    main()
