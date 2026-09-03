#!/usr/bin/env python3
"""VAE + IMU-only critic. A separate experimental track - GridReachModel
(176 px, runs/grid_leadwindow_emg_imu) is untouched by this file existing.

    python scripts/train_vae_discriminator_model.py \\
        --root "/media/.../emg_imu_vive" \\
        --config configs/tracked_grid_vae_discriminator.yaml \\
        --cache-dir artifacts/tracked_cache_posture \\
        --device cuda --epochs 20 --lead-window-ms 50 400 \\
        --output-dir runs/vae_discriminator_emg_imu

See src/emg_touch/models/vae_discriminator.py's module docstring for the
design of both pieces - in particular why the "discriminator" is a
one-directional IMU-only critic with a hinge loss rather than a gradient-
-reversal discriminator (the naive GRL version trains the encoder to make
EMG's presence UNDETECTABLE in the latent, the opposite of the goal).

Reports four numbers every epoch, not one, because a single averaged score
would hide exactly the thing this run exists to measure:

  main px      the full EMG+IMU prediction's error - comparable to every
               other number this project has reported.
  critic px    what IMU ALONE can do, from the same trunk, at its own best.
  margin       main px BETTER than critic px (positive = EMG is earning its
               keep by this measure; this is the quantity emg_importance_
               margin_loss directly optimises).
  KL           should stay small and bounded, not run away - a repeat of
               the earlier sigma-drift failure would show here first.

This is exploratory: it may not beat 176 px, and un-doing the exact
simplification that produced 176 px (one objective, no VAE, no adversarial
split) is expected to cost some accuracy in exchange for whatever the
margin and KL terms buy. Both are worth knowing, which is why every number
above is reported rather than only the final point error.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    build_tracked_loaders,
    emg_feature_count,
    imu_feature_count,
)
from emg_touch.grid_training import grid_point_loss  # noqa: E402
from emg_touch.models.vae_discriminator import (  # noqa: E402
    EMGImportanceVAE,
    emg_importance_margin_loss,
    vae_kl_loss,
)
from emg_touch.utils import choose_device, save_json, seed_everything  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_grid", Path(__file__).resolve().parent / "train_grid_reach_model.py"
)
_grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_grid)
make_grid_window = _grid.make_grid_window
canvas_from_disk = _grid.canvas_from_disk


def lead_window_from_ms(config: dict, leads_ms):
    if not leads_ms:
        return None
    rate = float(config["data"]["sample_rate_hz"]) / max(
        1, int(config["data"].get("decimation", 10))
    )
    low, high = sorted(leads_ms)
    return (max(1, int(round(low * rate / 1000.0))), max(2, int(round(high * rate / 1000.0))))


def compute_losses(model, window, config) -> dict:
    outputs = model(window["emg"], window["imu"], window["time_mask"])
    main_losses = grid_point_loss(outputs, window, config)
    # outputs["critic"] is self-contained (its own candidates/probabilities/
    # offsets from critic_decoded) - see the model's forward() for why that
    # matters here.
    critic_losses = grid_point_loss(outputs["critic"], window, config)

    kl = vae_kl_loss(outputs["mu_z"], outputs["log_var_z"])
    loss_config = config.get("loss", {})
    margin = float(loss_config.get("emg_margin", 0.02))
    hinge = emg_importance_margin_loss(
        outputs["prediction"], outputs["critic_prediction"], window["target"], margin
    )

    kl_weight = float(loss_config.get("kl_weight", 1e-4))
    critic_weight = float(loss_config.get("critic_weight", 1.0))
    margin_weight = float(loss_config.get("margin_weight", 0.5))
    total = (
        main_losses["loss"]
        + critic_weight * critic_losses["loss"]
        + kl_weight * kl
        + margin_weight * hinge
    )
    return {
        "loss": total, "main_loss": main_losses["loss"].detach(),
        "critic_loss": critic_losses["loss"].detach(), "kl": kl.detach(),
        "hinge": hinge.detach(), "outputs": outputs,
    }


@torch.no_grad()
def evaluate(model, loader, config, minimum_prefix, patch_length, lead_window,
             canvas_tensor, device) -> dict:
    model.eval()
    generator = np.random.default_rng(0)
    main_px, critic_px, kl_values = [], [], []
    for batch in loader:
        if batch is None:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        window = make_grid_window(
            batch, minimum_prefix, patch_length, generator, (), cutoffs_per_trial=1,
            fallback_canvas=canvas_tensor, lead_window=lead_window,
        )
        if window is None:
            continue
        outputs = model(window["emg"], window["imu"], window["time_mask"])
        target = window["target"]
        canvas = window.get("canvas_size")
        if canvas is None:
            continue
        main_px.extend(((outputs["prediction"] - target) * canvas).norm(dim=-1).cpu().tolist())
        critic_px.extend(
            ((outputs["critic_prediction"] - target) * canvas).norm(dim=-1).cpu().tolist()
        )
        kl_values.append(float(vae_kl_loss(outputs["mu_z"], outputs["log_var_z"])))
    if not main_px:
        return {}
    return {
        "main_px": float(np.mean(main_px)),
        "critic_px": float(np.mean(critic_px)),
        "margin_px": float(np.mean(critic_px) - np.mean(main_px)),
        "kl": float(np.mean(kl_values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/tracked_grid_vae_discriminator.yaml")
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache")
    parser.add_argument("--output-dir", default="runs/vae_discriminator")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cutoffs-per-trial", type=int, default=3)
    parser.add_argument("--lead-window-ms", type=float, nargs=2, metavar=("MIN", "MAX"))
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    seed_everything(int(args.seed if args.seed is not None else config.get("seed", 42)))
    device = choose_device(args.device)

    train_loader, validation_loader, test_loader = build_tracked_loaders(
        config, args.root, Path(args.cache_dir)
    )
    lead_window = lead_window_from_ms(
        config, tuple(args.lead_window_ms) if args.lead_window_ms else None
    )
    if lead_window:
        print(f"lead window {lead_window[0]}-{lead_window[1]} samples")

    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = EMGImportanceVAE(config, emg_channels, imu_channels).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters "
          f"(latent_dim={model.latent_dim})")

    minimum_prefix = int(config["virtual_leader"]["minimum_prefix"])
    patch_length = int(config["model"]["patch_length"])
    fallback = canvas_from_disk(args.root)
    if fallback:
        print(f"canvas {fallback[0]:.0f} x {fallback[1]:.0f} px")
    canvas_tensor = (
        torch.tensor(fallback, dtype=torch.float32, device=device) if fallback else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=float(config["training"].get("scheduler_factor", 0.5)),
        patience=int(config["training"].get("scheduler_patience", 3)),
    )
    generator = np.random.default_rng(int(args.seed or 0))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best, best_state, history = float("inf"), None, []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        running = {"loss": [], "main": [], "critic": [], "kl": [], "hinge": []}
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            if batch is None:
                continue
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            window = make_grid_window(
                batch, minimum_prefix, patch_length, generator, (),
                args.cutoffs_per_trial, fallback_canvas=canvas_tensor,
                lead_window=lead_window,
            )
            if window is None:
                continue
            result = compute_losses(model, window, config)
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            running["loss"].append(result["loss"].item())
            running["main"].append(result["main_loss"].item())
            running["critic"].append(result["critic_loss"].item())
            running["kl"].append(result["kl"].item())
            running["hinge"].append(result["hinge"].item())

        scores = evaluate(
            model, validation_loader, config, minimum_prefix, patch_length,
            lead_window, canvas_tensor, device,
        )
        scheduler.step(scores.get("main_px", 1e9))
        history.append({"epoch": epoch, **{f"train_{k}": float(np.mean(v or [0]))
                                            for k, v in running.items()}, **scores})
        print(
            f"epoch={epoch} main={np.mean(running['main'] or [0]):.3f} "
            f"critic={np.mean(running['critic'] or [0]):.3f} "
            f"kl={np.mean(running['kl'] or [0]):.3f} "
            f"hinge={np.mean(running['hinge'] or [0]):.3f} | "
            f"val main={scores.get('main_px', float('nan')):.1f}px "
            f"critic={scores.get('critic_px', float('nan')):.1f}px "
            f"margin={scores.get('margin_px', float('nan')):+.1f}px "
            f"kl={scores.get('kl', float('nan')):.3f}"
        )
        if scores.get("main_px", float("inf")) < best:
            best = scores["main_px"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state": best_state, "config": config}, output / "best.pt")

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(
        model, test_loader, config, minimum_prefix, patch_length, lead_window,
        canvas_tensor, device,
    )
    print("\n=== test ===")
    print(f"  main   (EMG+IMU) : {test.get('main_px', float('nan')):.1f} px")
    print(f"  critic (IMU only): {test.get('critic_px', float('nan')):.1f} px")
    print(f"  margin (EMG's measured contribution): {test.get('margin_px', float('nan')):+.1f} px")
    print(f"  KL: {test.get('kl', float('nan')):.3f}")
    print(f"\n  for reference, the proven deterministic pipeline reaches 176.3 px -")
    print(f"  compare main_px to that, not to critic_px, to judge this run overall.")

    save_json({"history": history, "test": test}, output / "results.json")
    print(f"\nwrote {output / 'results.json'}")


if __name__ == "__main__":
    main()
