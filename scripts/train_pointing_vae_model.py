#!/usr/bin/env python3
"""Train PointingBottleneckModel: one objective, no sampling, no adversarial split.

Deliberate retreat from PointingIntentVAE (VAE sampling + a GRL-trained
kinematic/anticipatory/residual split on top of grid_point_loss), after that
model reached 426 px - worse than the plain deterministic GridReachModel's
406 px, itself worse than a smaller GRU's 392.7 px. Three architectures,
ranked exactly inversely with how much machinery each stacked onto the same
encoder. See src/emg_touch/models/grid_reach.py's PointingBottleneckModel
docstring for the mechanism: sigma drifted from its 0.135 init toward the
N(0,I) prior's 1.0 instead of staying informative, which is what KL's own
gradient does whenever the reconstruction loss does not want a precise z
badly enough to fight it - and every training step then injected that
drifting noise into the point head on top of GRL adversarial pressure and
anticipatory-subspace dropout.

    python scripts/train_pointing_vae_model.py \\
        --root "/media/.../emg_imu_vive" \\
        --config configs/tracked_grid_reach.yaml \\
        --cache-dir artifacts/tracked_cache \\
        --device cuda --epochs 20 \\
        --inputs emg+imu --output-dir runs/pointing_bottleneck_emg_imu

What changed from the VAE version:

  ONE objective. total = grid_point_loss's own loss, full stop - no KL, no
  kinematic_predict/kinematic_adversarial/session_adversarial added to it.
  Those terms are gone from the model entirely (not just unweighted),
  because with nothing training the kinematic/anticipatory split to mean
  anything, keeping it around would be dead capacity pretending to be
  interpretable.

  No IMU-related dropout. The only dropout-like mechanism the predecessor
  had was anticipatory_dropout (zeroing part of the latent on 10% of
  training steps) - there is no IMU-modality dropout anywhere in this
  encoder to begin with (confirmed by grep before assuming otherwise), so
  this is the mechanism actually being removed. Standard transformer
  dropout (model.dropout, applied identically inside both the EMG and IMU
  branches) is untouched - that is ordinary architecture regularisation,
  not the noise-injection mechanism diagnosed here.

  No sampling. Without a KL term there is nothing to oppose injected
  z = mu + sigma*eps noise, so sampling would just be pure noise with no
  regularising purpose - removing it is the correct consequence of
  removing KL, not an independent change.

  A ReduceLROnPlateau scheduler, stepped on validation pixel error. This
  targets the specific failure signature the VAE run showed directly:
  training loss falling every epoch while validation error plateaued and
  oscillated (380-420 px for the last 15 of 20 epochs) - a fixed learning
  rate kept taking steps of the same size after progress had already
  stalled. This is this project's own precedent (configs/hill_fusion.yaml,
  the config that reached 184 px, uses the same scheduler_patience=3,
  scheduler_factor=0.5), not a new guess.

One thing this loses, stated plainly rather than hidden: the EMG-unique-
-contribution measurement (silencing z_anticipatory) that the VAE version
produced. Nothing here trains a subspace to hold EMG's surplus specifically,
so there is no principled subspace left to silence. If EMG turns out to
matter after IMU is disentangled at the mechanism level, that measurement
is worth rebuilding - not before, on a signal three independent
measurements now put at or below noise.
"""
from __future__ import annotations

import argparse
import inspect
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
)
from emg_touch.grid_training import grid_point_loss  # noqa: E402
from emg_touch.models.grid_reach import PointingBottleneckModel  # noqa: E402
from emg_touch.utils import save_json, seed_everything  # noqa: E402


def canvas_from_disk(root: str) -> tuple[float, float] | None:
    """Read the canvas size from one trial CSV, bypassing the trial cache.

    Duplicated rather than imported - see train_grid_reach_model.py's copy
    of this same function for why a tiny stateless utility is preferred over
    a cross-script import here.
    """
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


def make_pointing_window(
    batch: dict, minimum_prefix: int, patch_length: int, generator,
    ablate: tuple[str, ...] = (), cutoffs_per_trial: int = 1,
    fallback_canvas: torch.Tensor | None = None,
) -> dict | None:
    """Cut after onset. The tracker never appears here at all any more -
    the predecessor's label-only velocity/acceleration existed solely to
    supervise the kinematic subspace, which this model does not have.
    """
    lengths, onsets = batch["lengths"], batch["onset"]
    chosen = []
    for row in range(len(lengths)):
        length = int(lengths[row])
        touch = length - 1
        start = max(int(onsets[row]) + minimum_prefix, minimum_prefix)
        latest = touch - minimum_prefix
        if latest <= start:
            continue
        for _ in range(cutoffs_per_trial):
            cut = int(generator.integers(start, latest))
            progress = (cut - start) / max(latest - start, 1)
            chosen.append((row, cut, progress))
    if not chosen:
        return None

    prefix_length = max(
        patch_length, min(minimum_prefix * 4, min(cut for _, cut, _ in chosen))
    )
    device = batch["position"].device
    window: dict[str, torch.Tensor] = {}
    for key in ("emg", "imu"):
        window[key] = torch.stack(
            [batch[key][row, cut - prefix_length : cut] for row, cut, _ in chosen]
        )
    for name in ablate:
        if name in window:
            window[name] = torch.zeros_like(window[name])
    window["time_mask"] = torch.ones(
        len(chosen), prefix_length, dtype=torch.bool, device=device
    )

    rows = torch.tensor([row for row, _, _ in chosen], dtype=torch.long, device=device)
    window["target"] = batch["screen_target"].index_select(0, rows)
    if "canvas" in batch:
        window["canvas_size"] = batch["canvas"].index_select(0, rows)
    elif fallback_canvas is not None:
        window["canvas_size"] = fallback_canvas.to(device).unsqueeze(0).expand(len(rows), -1)

    minimum_weight = 0.5
    progress = torch.tensor([p for _, _, p in chosen], dtype=torch.float32, device=device)
    window["loss_weight"] = minimum_weight + (1.0 - minimum_weight) * progress
    return window


@torch.no_grad()
def evaluate(
    model, loader, config, device, minimum_prefix, patch_length, ablate,
    canvas_tensor, mean_target,
) -> dict:
    model.eval()
    totals: dict[str, list[float]] = {}
    generator = np.random.default_rng(0)
    for batch in loader:
        if batch is None:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        window = make_pointing_window(
            batch, minimum_prefix, patch_length, generator, ablate,
            cutoffs_per_trial=1, fallback_canvas=canvas_tensor,
        )
        if window is None:
            continue
        outputs = model(window["emg"], window["imu"], window["time_mask"])
        target = window["target"]
        canvas = window.get("canvas_size")

        candidates = {
            "direct": outputs["prediction"],
            "grid": outputs.get("grid_prediction", outputs["prediction"]),
            "mean": mean_target.to(device).unsqueeze(0).expand_as(target),
            "centre": torch.full_like(target, 0.5),
        }
        for name, prediction in candidates.items():
            delta = prediction - target
            totals.setdefault(f"{name}_norm", []).append(float(delta.norm(dim=-1).mean()))
            if canvas is not None:
                pixels = (delta * canvas).norm(dim=-1)
                totals.setdefault(f"{name}_px", []).append(float(pixels.mean()))
    return {key: float(np.mean(values)) for key, values in totals.items()}


def training_mean_target(loader) -> torch.Tensor:
    collected = []
    for batch in loader:
        if batch is not None and "screen_target" in batch:
            collected.append(batch["screen_target"].mean(0))
    if not collected:
        return torch.tensor([0.5, 0.5])
    return torch.stack(collected).mean(0).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/tracked_grid_reach.yaml")
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache")
    parser.add_argument("--output-dir", default="runs/pointing_bottleneck")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cutoffs-per-trial", type=int, default=3)
    parser.add_argument("--holdout-config")
    parser.add_argument(
        "--inputs", choices=("emg", "emg+imu"), default="emg+imu",
        help="emg: the muscle signal alone, no IMU encoder built. The "
        "tracker never enters the encoder in either case - see the "
        "structural check printed at startup.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.holdout_config:
        config.setdefault("data", {})["holdout_config"] = args.holdout_config
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    seed_everything(int(args.seed if args.seed is not None else config.get("seed", 42)))
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu"
    )

    train_loader, validation_loader, test_loader = build_tracked_loaders(
        config, args.root, Path(args.cache_dir)
    )
    use_imu = args.inputs == "emg+imu"
    ablate = () if use_imu else ("imu",)
    print(f"inputs: {args.inputs}  (IMU encoder {'built' if use_imu else 'not built'})")
    print(
        f"train {len(train_loader.dataset)} | val {len(validation_loader.dataset)} "
        f"| test {len(test_loader.dataset)} trials"
    )

    minimum_prefix = int(config["virtual_leader"]["minimum_prefix"])
    patch_length = int(config["model"]["patch_length"])
    if minimum_prefix < patch_length:
        raise ValueError(
            f"virtual_leader.minimum_prefix ({minimum_prefix}) must be >= "
            f"model.patch_length ({patch_length})"
        )

    emg_channels = emg_feature_count(config["data"])
    imu_channels = 6 * len(config["data"].get("sensors", ["S0", "S4", "S8", "S12"]))
    model = PointingBottleneckModel(config, emg_channels, imu_channels, use_imu).to(device)

    # Structural tracker-blindness check, on a real batch from the real
    # loader - two independent facts, not one assertion trusted on faith.
    forward_params = set(inspect.signature(PointingBottleneckModel.forward).parameters)
    tracker_params = forward_params & {"position", "velocity", "acceleration"}
    assert not tracker_params, (
        f"model.forward accepts tracker-shaped parameters: {tracker_params}"
    )
    probe_batch = next(iter(train_loader))
    probe_batch = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in probe_batch.items()
    }
    probe_window = make_pointing_window(
        probe_batch, minimum_prefix, patch_length, np.random.default_rng(0), ablate
    )
    assert not ({"position", "velocity"} & set(probe_window)), (
        "position/velocity present in the window dict under their own names"
    )
    print(
        "tracker-blindness check: model.forward's signature has no position/"
        "velocity/acceleration parameter, and the window built for it "
        "carries no such keys at all - the tracker is not used anywhere in "
        "this script, not even as a label"
    )

    fallback_canvas = canvas_from_disk(args.root)
    if fallback_canvas:
        print(f"canvas {fallback_canvas[0]:.0f} x {fallback_canvas[1]:.0f} px")
    canvas_tensor = (
        torch.tensor(fallback_canvas, dtype=torch.float32, device=device)
        if fallback_canvas else None
    )
    mean_target = training_mean_target(train_loader)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    # Same schedule as configs/hill_fusion.yaml, the config that reached
    # 184 px - not a new guess. Targets the exact failure signature the
    # predecessor showed: training loss falling every epoch while
    # validation error plateaued/oscillated, i.e. the fixed step size was
    # too large once real progress had stalled.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=float(config["training"].get("scheduler_factor", 0.5)),
        patience=int(config["training"].get("scheduler_patience", 3)),
        min_lr=float(config["training"].get("minimum_learning_rate", 1e-6)),
    )
    generator = np.random.default_rng(int(args.seed or 0))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best, best_state, history = float("inf"), None, []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        running: list[float] = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            if batch is None:
                continue
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            window = make_pointing_window(
                batch, minimum_prefix, patch_length, generator, ablate,
                args.cutoffs_per_trial, canvas_tensor,
            )
            if window is None:
                continue
            outputs = model(window["emg"], window["imu"], window["time_mask"])
            # ONE objective: grid_point_loss's own total. Nothing else is
            # added to the graph.
            losses = grid_point_loss(outputs, window, config)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            running.append(float(losses["loss"].detach()))

        scores = evaluate(
            model, validation_loader, config, device, minimum_prefix, patch_length,
            ablate, canvas_tensor, mean_target,
        )
        selection = scores.get("direct_px", scores.get("direct_norm", 1e9))
        scheduler.step(selection)
        history.append({"epoch": epoch, "train": float(np.mean(running or [0])), **scores})
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch} loss={np.mean(running or [0]):.4f} lr={current_lr:.2e} | "
            f"val direct={scores.get('direct_px', float('nan')):.1f}px "
            f"grid={scores.get('grid_px', float('nan')):.1f}px"
        )
        if selection < best:
            best = selection
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, output / "best.pt")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"loaded best checkpoint (val {best:.1f} px) for test")
    test = evaluate(
        model, test_loader, config, device, minimum_prefix, patch_length,
        ablate, canvas_tensor, mean_target,
    )

    print("\n=== test ===")
    print(f"  inputs: {args.inputs}, tracker excluded from the encoder\n")
    print(f"  {'':10}{'norm':>10}{'pixels':>12}")
    for name in ("direct", "grid", "mean", "centre"):
        if f"{name}_norm" not in test:
            continue
        px = f"{test[f'{name}_px']:.1f}" if f"{name}_px" in test else "n/a"
        print(f"  {name:10}{test[f'{name}_norm']:>10.4f}{px:>12}")

    if "direct_px" in test and "mean_px" in test:
        gain = (test["mean_px"] - test["direct_px"]) / test["mean_px"] * 100
        below_200 = "YES" if test["direct_px"] < 200 else "no"
        print(f"\n  direct prediction is {gain:+.1f}% vs the population mean target")
        print(f"  below 200 px: {below_200} ({test['direct_px']:.1f} px)")

    save_json({"history": history, "test": test}, output / "results.json")
    print(f"\nwrote {output / 'results.json'}")


if __name__ == "__main__":
    main()
