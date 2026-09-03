#!/usr/bin/env python3
"""Add (mu, sigma) to a proven, frozen GridReachModel. No sampling, no KL.

    python scripts/train_uncertainty_head.py \\
        --root "/media/.../emg_imu_vive" \\
        --checkpoint runs/grid_leadwindow_emg_imu/best.pt \\
        --config configs/tracked_grid_within.yaml \\
        --cache-dir artifacts/tracked_cache_posture \\
        --device cuda --epochs 15 --inputs emg+imu \\
        --lead-window-ms 50 400 \\
        --output-dir runs/uncertainty_head

Loads --checkpoint FROZEN (requires_grad_(False), permanently in eval mode)
and trains only a small new UncertaintyHead (~2K parameters) on top of it -
see src/emg_touch/models/grid_reach.py's UncertaintyHead docstring for why
this is a heteroscedastic output head rather than a sampled-latent VAE. The
176 px point model never receives a gradient here; this script cannot make
it worse, only add a confidence radius to what it already predicts.

Reports two things, not one: NLL is a fit statistic and can look good while
being wrong. Coverage is the actual test - among held-out points, what
fraction of true targets fall within the predicted 1-sigma and 2-sigma
ellipse. A well-calibrated Gaussian should show ~68% and ~95%. Systematically
low coverage means sigma is too small (overconfident, the dangerous
direction for a UI that acts on it); systematically high means it is too
large (safe but uninformative).
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
from emg_touch.models.grid_reach import (  # noqa: E402
    GridReachModel,
    UncertaintyHead,
    gaussian_nll_loss,
)
from emg_touch.utils import choose_device, save_json, seed_everything  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_grid", Path(__file__).resolve().parent / "train_grid_reach_model.py"
)
_grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_grid)
make_grid_window = _grid.make_grid_window
canvas_from_disk = _grid.canvas_from_disk


def lead_window_from_ms(config: dict, leads_ms: tuple[float, float] | None):
    if not leads_ms:
        return None
    rate = float(config["data"]["sample_rate_hz"]) / max(
        1, int(config["data"].get("decimation", 10))
    )
    low, high = sorted(leads_ms)
    return (max(1, int(round(low * rate / 1000.0))), max(2, int(round(high * rate / 1000.0))))


@torch.no_grad()
def evaluate(
    base, head, loader, minimum_prefix, patch_length, ablate, lead_window,
    canvas_tensor, device,
) -> dict:
    head.eval()
    generator = np.random.default_rng(0)
    residual_sigma_ratio = []  # |target - mu| / sigma, per axis
    nll_values, px_errors = [], []
    for batch in loader:
        if batch is None:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        window = make_grid_window(
            batch, minimum_prefix, patch_length, generator, ablate,
            cutoffs_per_trial=1, fallback_canvas=canvas_tensor, lead_window=lead_window,
        )
        if window is None:
            continue
        outputs = base(window["emg"], window["imu"], window["time_mask"])
        mu = outputs["prediction"]
        sigma = head(outputs["context"])
        target = window["target"]
        canvas = window.get("canvas_size")

        nll_values.append(float(gaussian_nll_loss(mu, sigma, target)))
        residual_sigma_ratio.append(((target - mu).abs() / sigma).cpu().numpy())
        if canvas is not None:
            px_errors.append(((mu - target) * canvas).norm(dim=-1).cpu().numpy())

    if not nll_values:
        return {}
    ratio = np.concatenate(residual_sigma_ratio, axis=0)  # (N, 2) per-axis z-scores
    # Coverage for an ISOTROPIC 2-D Gaussian is governed by the radial
    # distance in units of sigma (chi with 2 dof), not the per-axis ratio
    # directly - but sigma is fit per axis here, so report per-axis coverage
    # against the 1-D normal thresholds, which is what each axis's own NLL
    # actually optimises for and is the honest thing to check against it.
    scores = {
        "nll": float(np.mean(nll_values)),
        "coverage_1sigma": float(np.mean(ratio <= 1.0)),
        "coverage_2sigma": float(np.mean(ratio <= 2.0)),
        "mean_sigma_px": None,
    }
    if px_errors:
        scores["direct_px"] = float(np.mean(np.concatenate(px_errors)))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache")
    parser.add_argument("--output-dir", default="runs/uncertainty_head")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cutoffs-per-trial", type=int, default=3)
    parser.add_argument("--inputs", choices=("emg", "emg+imu"), default="emg+imu")
    parser.add_argument(
        "--lead-window-ms", type=float, nargs=2, metavar=("MIN", "MAX"),
        help="Should match the lead window the base checkpoint was trained "
        "with - the uncertainty head is calibrated to whatever regime it "
        "sees examples from.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(args.seed if args.seed is not None else config.get("seed", 42)))
    device = choose_device(args.device)

    train_loader, validation_loader, test_loader = build_tracked_loaders(
        config, args.root, Path(args.cache_dir)
    )
    use_imu = args.inputs == "emg+imu"
    ablate = () if use_imu else ("imu",)
    lead_window = lead_window_from_ms(config, tuple(args.lead_window_ms) if args.lead_window_ms else None)
    if lead_window:
        print(f"lead window {lead_window[0]}-{lead_window[1]} samples")

    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    base = GridReachModel(config, emg_channels, imu_channels, use_imu).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    base.load_state_dict(state["model_state"] if "model_state" in state else state)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    print(f"base model frozen: {sum(p.numel() for p in base.parameters()):,} parameters, "
          "0 trainable")

    context_dim = base.head.shared[1].in_features if hasattr(base.head, "shared") else (
        int(config["model"]["d_model"]) * (2 if use_imu else 1)
    )
    head = UncertaintyHead(context_dim).to(device)
    print(f"uncertainty head: {sum(p.numel() for p in head.parameters()):,} trainable "
          "parameters\n")

    minimum_prefix = int(config["virtual_leader"]["minimum_prefix"])
    patch_length = int(config["model"]["patch_length"])
    fallback = canvas_from_disk(args.root)
    canvas_tensor = (
        torch.tensor(fallback, dtype=torch.float32, device=device) if fallback else None
    )

    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    generator = np.random.default_rng(int(args.seed or 0))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_nll, best_state = float("inf"), None
    history = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        running = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            if batch is None:
                continue
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            window = make_grid_window(
                batch, minimum_prefix, patch_length, generator, ablate,
                args.cutoffs_per_trial, fallback_canvas=canvas_tensor, lead_window=lead_window,
            )
            if window is None:
                continue
            with torch.no_grad():
                outputs = base(window["emg"], window["imu"], window["time_mask"])
            sigma = head(outputs["context"])
            loss = gaussian_nll_loss(outputs["prediction"], sigma, window["target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running.append(loss.item())

        scores = evaluate(
            base, head, validation_loader, minimum_prefix, patch_length, ablate,
            lead_window, canvas_tensor, device,
        )
        history.append({"epoch": epoch, "train_nll": float(np.mean(running or [0])), **scores})
        print(
            f"epoch={epoch} train_nll={np.mean(running or [0]):.4f} | "
            f"val_nll={scores.get('nll', float('nan')):.4f} "
            f"coverage_1sig={scores.get('coverage_1sigma', float('nan')) * 100:.1f}% "
            f"coverage_2sig={scores.get('coverage_2sigma', float('nan')) * 100:.1f}%"
        )
        if scores.get("nll", float("inf")) < best_nll:
            best_nll = scores["nll"]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            torch.save(
                {"head_state": best_state, "base_checkpoint": str(args.checkpoint), "config": config},
                output / "best.pt",
            )

    if best_state is not None:
        head.load_state_dict(best_state)
    test = evaluate(
        base, head, test_loader, minimum_prefix, patch_length, ablate,
        lead_window, canvas_tensor, device,
    )
    print("\n=== test ===")
    print(f"  NLL              : {test.get('nll', float('nan')):.4f}")
    print(f"  coverage @ 1sigma: {test.get('coverage_1sigma', float('nan')) * 100:.1f}%  "
          f"(want ~68% - well-calibrated Gaussian)")
    print(f"  coverage @ 2sigma: {test.get('coverage_2sigma', float('nan')) * 100:.1f}%  "
          f"(want ~95%)")
    if "direct_px" in test:
        print(f"  point error (unchanged from the frozen base): {test['direct_px']:.1f} px")
    low1 = test.get("coverage_1sigma", 0.68) < 0.60
    low2 = test.get("coverage_2sigma", 0.95) < 0.88
    if low1 or low2:
        print("\n  Coverage below target: sigma is OVERCONFIDENT. Treat predicted")
        print("  confidence radii as optimistic until this is addressed - do not")
        print("  wire this into anything that acts on the interval as-is.")

    save_json({"history": history, "test": test}, output / "results.json")
    print(f"\nwrote {output / 'results.json'}")


if __name__ == "__main__":
    main()
