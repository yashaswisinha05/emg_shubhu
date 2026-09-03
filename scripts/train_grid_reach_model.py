#!/usr/bin/env python3
"""Train the grid+offset screen-touch model, aiming below 200 px.

The plain reach-target model (train_reach_target_model.py) scored 329-460 px
per condition with a GRU and direct regression. This borrows the two
architectural choices that reached 184 px on this project's original
screen-touch dataset - a patch transformer and a grid+offset head - see
src/emg_touch/models/grid_reach.py for what is reused unmodified (the loss)
and what is deliberately simplified (a single fused encoder rather than a
residual-onto-pretrained-IMU scheme this dataset has no pretraining run for
yet).

    python scripts/train_grid_reach_model.py \\
        --root "/media/.../emg_imu_vive" \\
        --config configs/tracked_grid_reach.yaml \\
        --cache-dir artifacts/tracked_cache \\
        --device cuda --epochs 20 \\
        --inputs emg+imu --output-dir runs/grid_reach_emg_imu

The tracker is never an input, exactly as in every other wearable-mode
script in this project - verified structurally in tests, not just asserted.

--cutoffs-per-trial (default 3) draws that many independent cutoffs per
trial per training step, matching the original pipeline's continual-cutoff
training rather than relying on epoch-to-epoch resampling alone to cover the
space of elapsed times a real deployment would see. Later cutoffs are
weighted more heavily (loss_weight rises with progress toward the touch),
the same shape the original continual training used and for the same
reason: an early cutoff is close to guessing by construction, and weighting
it equally to a near-touch cutoff would blur the gradient with examples the
model cannot yet be expected to get right.
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
from emg_touch.models.grid_reach import GridReachModel  # noqa: E402
from emg_touch.utils import save_json, seed_everything  # noqa: E402


def canvas_from_disk(root: str) -> tuple[float, float] | None:
    """Read the canvas size from one trial CSV, bypassing the trial cache.

    Duplicated from train_reach_target_model.py rather than imported: it is
    fifteen lines with no state, and importing across sibling scripts here
    has already caused one accidental cross-script coupling in this project
    (TrajectoryEncoder reuse) that a device-mismatch bug then had to be
    chased through. A tiny duplicated utility is the cheaper failure mode.
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


def make_grid_window(
    batch: dict, minimum_prefix: int, patch_length: int, generator,
    ablate: tuple[str, ...] = (), cutoffs_per_trial: int = 1,
    fallback_canvas: torch.Tensor | None = None,
) -> dict | None:
    """Cut after onset; each row can contribute several independent cutoffs.

    Returns a single dict usable directly as grid_point_loss's `batch`
    argument (it carries `target`, `canvas_size`, `loss_weight`) as well as
    the model's own inputs (`emg`, `imu`, `time_mask`).
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
            chosen.append((row, cut, touch, progress))
    if not chosen:
        return None

    # PatchTransformerEncoder needs at least one full patch.
    prefix_length = max(
        patch_length, min(minimum_prefix * 4, min(cut for _, cut, _, _ in chosen))
    )
    device = batch["position"].device
    window: dict[str, torch.Tensor] = {}
    for key in ("emg", "imu"):
        window[key] = torch.stack(
            [batch[key][row, cut - prefix_length : cut] for row, cut, _, _ in chosen]
        )
    for name in ablate:
        if name in window:
            window[name] = torch.zeros_like(window[name])
    window["time_mask"] = torch.ones(
        len(chosen), prefix_length, dtype=torch.bool, device=device
    )

    rows = torch.tensor(
        [row for row, _, _, _ in chosen], dtype=torch.long, device=device
    )
    window["target"] = batch["screen_target"].index_select(0, rows)
    if "canvas" in batch:
        window["canvas_size"] = batch["canvas"].index_select(0, rows)
    elif fallback_canvas is not None:
        # grid_point_loss reads batch["canvas_size"] unconditionally (pixel,
        # radial and transport losses all scale by it), so a missing canvas
        # is not something training can silently skip the way an optional
        # metric can - it has to be supplied every time "canvas" is absent
        # from the batch, not only at evaluation. It is absent whenever the
        # trial was cached before canvas extraction existed in
        # preprocess_tracked_trial: the cache signature has no "canvas
        # support" marker, so an old cache entry is reused as-is and simply
        # lacks the field. First found and worked around in
        # train_reach_target_model.py; that fix covered evaluation there but
        # was never carried into this script's training loop, which is
        # exactly where grid_point_loss's requirement is unconditional and
        # the bug actually surfaces.
        window["canvas_size"] = fallback_canvas.to(rows.device).unsqueeze(0).expand(
            len(rows), -1
        )
    # A near-touch cutoff is a fair prediction task; a cutoff right after
    # onset is close to guessing by construction, and weighting the two
    # equally blurs the gradient with examples the model cannot yet be
    # expected to solve. Floor of 0.5 keeps early cutoffs contributing
    # rather than vanishing, matching the shape of the original pipeline's
    # continual-cutoff weighting.
    minimum_weight = 0.5
    progress = torch.tensor(
        [p for _, _, _, p in chosen], dtype=torch.float32, device=device
    )
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
        window = make_grid_window(
            batch, minimum_prefix, patch_length, generator, ablate,
            cutoffs_per_trial=1, fallback_canvas=canvas_tensor,
        )
        if window is None:
            continue
        canvas = window.get("canvas_size")
        outputs = model(window["emg"], window["imu"], window["time_mask"])
        target = window["target"]

        for name, prediction in (
            ("direct", outputs["prediction"]),
            ("grid", outputs.get("grid_prediction", outputs["prediction"])),
            ("mean", mean_target.to(device).unsqueeze(0).expand_as(target)),
            ("centre", torch.full_like(target, 0.5)),
        ):
            delta = prediction - target
            totals.setdefault(f"{name}_norm", []).append(
                float(delta.norm(dim=-1).mean())
            )
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
    parser.add_argument("--output-dir", default="runs/grid_reach")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cutoffs-per-trial", type=int, default=3)
    parser.add_argument("--holdout-config")
    parser.add_argument(
        "--inputs", choices=("emg", "emg+imu"), default="emg+imu",
        help="emg: the muscle signal alone, no separate IMU encoder is even "
        "built. The tracker never enters the encoder in either case.",
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
    print(f"inputs: {args.inputs}  (tracker never enters the encoder; "
          f"IMU encoder {'built' if use_imu else 'not built'})")
    print(
        f"train {len(train_loader.dataset)} | val {len(validation_loader.dataset)} "
        f"| test {len(test_loader.dataset)} trials"
    )

    minimum_prefix = int(config["virtual_leader"]["minimum_prefix"])
    patch_length = int(config["model"]["patch_length"])
    if minimum_prefix < patch_length:
        raise ValueError(
            f"virtual_leader.minimum_prefix ({minimum_prefix}) must be >= "
            f"model.patch_length ({patch_length}) - the encoder needs at "
            "least one full patch per window."
        )

    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = GridReachModel(config, emg_channels, imu_channels, use_imu).to(device)

    fallback_canvas = canvas_from_disk(args.root)
    if fallback_canvas:
        print(f"canvas {fallback_canvas[0]:.0f} x {fallback_canvas[1]:.0f} px")
    else:
        print("WARNING: no canvas found on disk, and any trial whose cache "
              "entry also lacks it will have no canvas_size at all - "
              "grid_point_loss will raise on that batch.")
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
            window = make_grid_window(
                batch, minimum_prefix, patch_length, generator, ablate,
                args.cutoffs_per_trial, fallback_canvas=canvas_tensor,
            )
            if window is None:
                continue
            outputs = model(window["emg"], window["imu"], window["time_mask"])
            losses = grid_point_loss(outputs, window, config)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            running.append(float(losses["loss"].detach()))

        scores = evaluate(
            model, validation_loader, config, device, minimum_prefix,
            patch_length, ablate, canvas_tensor, mean_target,
        )
        selection = scores.get("direct_px", scores.get("direct_norm", 1e9))
        history.append({"epoch": epoch, "train": float(np.mean(running or [0])), **scores})
        print(
            f"epoch={epoch} loss={np.mean(running or [0]):.4f} | "
            f"val direct={scores.get('direct_px', float('nan')):.1f} px "
            f"grid={scores.get('grid_px', float('nan')):.1f} px"
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
    print(f"  inputs: {args.inputs}, tracker excluded\n")
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
