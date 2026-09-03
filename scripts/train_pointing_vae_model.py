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
    imu_feature_count,
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
    context_samples: int | None = None,
    cutoff_offsets: tuple[int, ...] = (),
) -> dict | None:
    """Build fixed-history windows at random or onset-relative cutoffs.

    History duration and cutoff timing are independent. Earlier code used
    `minimum_prefix` for both and capped every prefix at 64 samples (~0.51 s),
    preventing a fair test of the 2.16 s context used by the pretrained
    emg2pose result. Missing pre-buffer history is left-zero-padded and marked
    false in `time_mask`; no future sample is fabricated.

    The tracker never appears here as a model input -
    the predecessor's label-only velocity/acceleration existed solely to
    supervise the kinematic subspace, which this model does not have.
    """
    lengths, onsets = batch["lengths"], batch["onset"]
    chosen = []
    for row in range(len(lengths)):
        length = int(lengths[row])
        touch = length - 1
        onset = int(onsets[row])
        start = max(onset + minimum_prefix, minimum_prefix)
        latest = touch - minimum_prefix
        if cutoff_offsets:
            if latest <= 0:
                continue
        elif latest <= start:
            continue
        for _ in range(cutoffs_per_trial):
            if cutoff_offsets:
                offset = int(generator.choice(cutoff_offsets))
                cut = onset + offset
                if cut <= 0 or cut > latest:
                    continue
            else:
                cut = int(generator.integers(start, latest))
                offset = cut - onset
            progress = float(np.clip(
                (cut - onset) / max(touch - onset, 1), 0.0, 1.0
            ))
            chosen.append((row, cut, progress, offset))
    if not chosen:
        return None

    prefix_length = max(patch_length, int(context_samples or minimum_prefix * 4))
    device = batch["emg"].device
    window: dict[str, torch.Tensor] = {}
    time_masks = []
    for key in ("emg", "imu"):
        slices = []
        for row, cut, _, _ in chosen:
            start_index = max(0, cut - prefix_length)
            available = batch[key][row, start_index:cut]
            padded = torch.zeros(
                prefix_length, batch[key].size(-1),
                dtype=batch[key].dtype, device=batch[key].device,
            )
            padded[-available.size(0):] = available
            slices.append(padded)
            if key == "emg":
                mask = torch.zeros(prefix_length, dtype=torch.bool, device=device)
                mask[-available.size(0):] = True
                time_masks.append(mask)
        window[key] = torch.stack(slices)
    for name in ablate:
        if name in window:
            window[name] = torch.zeros_like(window[name])
    window["time_mask"] = torch.stack(time_masks)

    rows = torch.tensor([row for row, _, _, _ in chosen], dtype=torch.long, device=device)
    window["target"] = batch["screen_target"].index_select(0, rows)
    if "canvas" in batch:
        window["canvas_size"] = batch["canvas"].index_select(0, rows)
    elif fallback_canvas is not None:
        window["canvas_size"] = fallback_canvas.to(device).unsqueeze(0).expand(len(rows), -1)

    minimum_weight = 0.5
    progress = torch.tensor(
        [p for _, _, p, _ in chosen], dtype=torch.float32, device=device
    )
    window["loss_weight"] = minimum_weight + (1.0 - minimum_weight) * progress
    window["samples_past_onset"] = torch.tensor(
        [offset for _, _, _, offset in chosen], dtype=torch.long, device=device
    )
    return window


@torch.no_grad()
def evaluate(
    model, loader, config, device, minimum_prefix, patch_length, ablate,
    canvas_tensor, mean_target, context_samples,
    cutoff_offsets: tuple[int, ...] = (),
) -> dict:
    model.eval()
    totals: dict[str, list[float]] = {}
    generator = np.random.default_rng(0)
    for batch in loader:
        if batch is None:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        # Every eligible trial is evaluated at every requested offset. One
        # random cutoff per trial made validation noisy and confounded timing
        # with trial identity.
        selected_offsets = tuple((offset,) for offset in cutoff_offsets) or ((),)
        for selected in selected_offsets:
            window = make_pointing_window(
                batch, minimum_prefix, patch_length, generator, ablate,
                cutoffs_per_trial=1, fallback_canvas=canvas_tensor,
                context_samples=context_samples, cutoff_offsets=selected,
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
            if bool(config.get("evaluation", {}).get("paired_interventions", False)):
                zero_emg = torch.zeros_like(window["emg"])
                candidates["without_emg"] = model(
                    zero_emg, window["imu"], window["time_mask"]
                )["prediction"]
                if window["emg"].size(0) > 1:
                    candidates["shuffled_emg"] = model(
                        torch.roll(window["emg"], 1, 0),
                        window["imu"], window["time_mask"],
                    )["prediction"]
                if getattr(model, "use_imu", False):
                    candidates["without_imu"] = model(
                        window["emg"], torch.zeros_like(window["imu"]),
                        window["time_mask"],
                    )["prediction"]

            offset = int(window["samples_past_onset"][0]) if selected else None
            for name, prediction in candidates.items():
                delta = prediction - target
                norms = delta.norm(dim=-1).detach().cpu().tolist()
                totals.setdefault(f"{name}_norm", []).extend(norms)
                if offset is not None:
                    totals.setdefault(
                        f"{name}_offset_{offset:+d}_norm", []
                    ).extend(norms)
                if canvas is not None:
                    pixels = (delta * canvas).norm(dim=-1).detach().cpu().tolist()
                    totals.setdefault(f"{name}_px", []).extend(pixels)
                    if offset is not None:
                        totals.setdefault(
                            f"{name}_offset_{offset:+d}_px", []
                        ).extend(pixels)
    return {key: float(np.mean(values)) for key, values in totals.items()}


def training_mean_target(loader) -> torch.Tensor:
    collected = []
    for batch in loader:
        if batch is not None and "screen_target" in batch:
            collected.extend(batch["screen_target"].unbind(0))
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
    parser.add_argument(
        "--context-ms", type=float,
        help="Fixed causal history duration. Missing history is left-padded "
        "and masked. Defaults to model.context_ms.",
    )
    parser.add_argument(
        "--cutoff-offset-ms", type=float, nargs="+",
        help="Cutoffs relative to tracker-defined movement onset. Training "
        "samples them and evaluation visits every offset for every trial.",
    )
    parser.add_argument(
        "--screen-only", action="store_true",
        help="Optimize direct pixel/radial error without grid auxiliaries.",
    )
    parser.add_argument(
        "--paired-interventions", action="store_true",
        help="Evaluate the same checkpoint with EMG removed/shuffled and IMU removed.",
    )
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
    if args.screen_only:
        config["loss"].update({
            "heatmap_weight": 0.0,
            "offset_weight": 0.0,
            "transport_weight": 0.0,
            "pixel_weight": 1.0,
            "radial_weight": 1.0,
        })
    if args.paired_interventions:
        config.setdefault("evaluation", {})["paired_interventions"] = True
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
    model_rate = float(config["data"]["sample_rate_hz"]) / max(
        1, int(config["data"].get("decimation", 10))
    )
    context_ms = float(
        args.context_ms
        if args.context_ms is not None
        else config["model"].get(
            "context_ms", minimum_prefix * 4 / model_rate * 1000.0
        )
    )
    context_samples = max(
        patch_length, int(round(context_ms * model_rate / 1000.0))
    )
    requested_offsets_ms = (
        args.cutoff_offset_ms
        if args.cutoff_offset_ms is not None
        else config.get("evaluation", {}).get("cutoff_offsets_ms", [])
    )
    cutoff_offsets = tuple(dict.fromkeys(
        int(round(float(value) * model_rate / 1000.0))
        for value in requested_offsets_ms
    ))
    print(
        f"context: {context_samples} samples "
        f"({context_samples / model_rate:.2f} s)"
    )
    if cutoff_offsets:
        print("cutoffs from movement onset: " + ", ".join(
            f"{offset / model_rate * 1000:+.0f} ms" for offset in cutoff_offsets
        ))

    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
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
        probe_batch, minimum_prefix, patch_length, np.random.default_rng(0), ablate,
        context_samples=context_samples, cutoff_offsets=cutoff_offsets,
    )
    assert not ({"position", "velocity"} & set(probe_window)), (
        "position/velocity present in the window dict under their own names"
    )
    print(
        "tracker-blindness check: model.forward's signature has no position/"
        "velocity/acceleration parameter, and the window built for it "
        "carries no such keys at all. Tracker-derived onset is used only to "
        "choose the evaluation time and the target only as supervision; "
        "neither can enter the prediction encoder."
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
                args.cutoffs_per_trial, canvas_tensor, context_samples,
                cutoff_offsets,
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
            ablate, canvas_tensor, mean_target, context_samples, cutoff_offsets,
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
        ablate, canvas_tensor, mean_target, context_samples, cutoff_offsets,
    )

    print("\n=== test ===")
    print(f"  inputs: {args.inputs}, tracker excluded from the encoder\n")
    print(f"  {'':10}{'norm':>10}{'pixels':>12}")
    for name in (
        "direct", "grid", "without_emg", "shuffled_emg", "without_imu",
        "mean", "centre",
    ):
        if f"{name}_norm" not in test:
            continue
        px = f"{test[f'{name}_px']:.1f}" if f"{name}_px" in test else "n/a"
        print(f"  {name:10}{test[f'{name}_norm']:>10.4f}{px:>12}")

    if cutoff_offsets:
        print("\n  direct error by cutoff from movement onset")
        for offset in cutoff_offsets:
            key = f"direct_offset_{offset:+d}_px"
            if key in test:
                print(f"    {offset / model_rate * 1000:+6.0f} ms: {test[key]:7.1f} px")

    if "without_emg_px" in test:
        full = test["direct_px"]
        print("\n  paired same-checkpoint effects (positive = modality helps)")
        print(f"    remove EMG: {test['without_emg_px'] - full:+.1f} px")
        if "shuffled_emg_px" in test:
            print(f"    shuffle EMG: {test['shuffled_emg_px'] - full:+.1f} px")
        if "without_imu_px" in test:
            print(f"    remove IMU: {test['without_imu_px'] - full:+.1f} px")

    if "direct_px" in test and "mean_px" in test:
        gain = (test["mean_px"] - test["direct_px"]) / test["mean_px"] * 100
        below_200 = "YES" if test["direct_px"] < 200 else "no"
        print(f"\n  direct prediction is {gain:+.1f}% vs the population mean target")
        print(f"  below 200 px: {below_200} ({test['direct_px']:.1f} px)")

    save_json({"history": history, "test": test}, output / "results.json")
    print(f"\nwrote {output / 'results.json'}")


if __name__ == "__main__":
    main()
