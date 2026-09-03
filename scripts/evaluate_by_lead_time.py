#!/usr/bin/env python3
"""Screen error as a function of how far ahead the prediction is made.

Every pointing number this project has reported - 390, 406, 426 px - is an
average over cutoffs drawn uniformly between movement onset and touch. That
average mixes two very different questions:

    "where will they touch?"  asked 1.2 s out, with the whole reach still to
                              come and almost nothing committed
    "where will they touch?"  asked 120 ms out, with the hand nearly there

and reports one number for both. If the error depends strongly on lead time
- and it must, since the second question is nearly answered by the arm's
current position - then the single average has been hiding a curve, and
"can we reach 200 px" has no single answer. It has an answer per lead time.

    python scripts/evaluate_by_lead_time.py \\
        --root "/media/.../emg_imu_vive" \\
        --checkpoint runs/grid_within_emg_imu/best.pt \\
        --config configs/tracked_grid_within.yaml \\
        --cache-dir artifacts/tracked_cache_posture \\
        --device cuda --inputs emg+imu

Cutoffs here are placed a fixed time BEFORE TOUCH, not after onset. Onset is
the natural axis for asking whether EMG anticipates movement; time-to-touch
is the natural axis for asking how much of the reach is left to forecast,
which is what governs this error. Both baselines are recomputed at every
lead time from the same trials, so a lead time that happens to include
easier trials is not credited for it.

This is evaluation only - it loads a checkpoint and never trains, so it
costs one pass over the test set and can be pointed at any existing run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    CANVAS_COLUMNS,
    build_tracked_loaders,
    emg_feature_count,
    imu_feature_count,
)
from emg_touch.models.grid_reach import (  # noqa: E402
    GridReachModel,
    PointingBottleneckModel,
)
from emg_touch.utils import choose_device, save_json  # noqa: E402


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
    batch: dict, lead_samples: int, minimum_prefix: int, patch_length: int,
    ablate: tuple[str, ...], fallback_canvas: torch.Tensor | None,
) -> dict | None:
    """One window per trial, cut exactly `lead_samples` before touch."""
    lengths, onsets = batch["lengths"], batch["onset"]
    chosen = []
    for row in range(len(lengths)):
        length = int(lengths[row])
        touch = length - 1
        cut = touch - lead_samples
        # The cut must sit after onset (a stationary arm has not committed to
        # anything yet) and leave a full patch of history behind it.
        if cut < max(int(onsets[row]) + minimum_prefix, patch_length):
            continue
        chosen.append((row, cut))
    if not chosen:
        return None

    prefix_length = max(patch_length, min(minimum_prefix * 4, min(c for _, c in chosen)))
    device = batch["position"].device
    window: dict[str, torch.Tensor] = {}
    for key in ("emg", "imu"):
        window[key] = torch.stack(
            [batch[key][row, cut - prefix_length : cut] for row, cut in chosen]
        )
    for name in ablate:
        if name in window:
            window[name] = torch.zeros_like(window[name])
    window["time_mask"] = torch.ones(
        len(chosen), prefix_length, dtype=torch.bool, device=device
    )
    rows = torch.tensor([row for row, _ in chosen], dtype=torch.long, device=device)
    window["target"] = batch["screen_target"].index_select(0, rows)
    if "canvas" in batch:
        window["canvas_size"] = batch["canvas"].index_select(0, rows)
    elif fallback_canvas is not None:
        window["canvas_size"] = fallback_canvas.to(device).unsqueeze(0).expand(len(rows), -1)
    return window


@torch.no_grad()
def evaluate_at_lead(
    model, loader, lead_samples, minimum_prefix, patch_length, ablate,
    canvas_tensor, mean_target, device,
) -> dict:
    model.eval()
    collected: dict[str, list[float]] = {}
    trials = 0
    for batch in loader:
        if batch is None:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        window = window_at_lead(
            batch, lead_samples, minimum_prefix, patch_length, ablate, canvas_tensor
        )
        if window is None:
            continue
        outputs = model(window["emg"], window["imu"], window["time_mask"])
        target = window["target"]
        canvas = window.get("canvas_size")
        if canvas is None:
            continue
        trials += len(target)
        for name, prediction in (
            ("direct", outputs["prediction"]),
            ("grid", outputs.get("grid_prediction", outputs["prediction"])),
            ("mean", mean_target.to(device).unsqueeze(0).expand_as(target)),
            ("centre", torch.full_like(target, 0.5)),
        ):
            pixels = ((prediction - target) * canvas).norm(dim=-1)
            collected.setdefault(name, []).extend(pixels.cpu().numpy().tolist())
    if not trials:
        return {}
    scores = {f"{k}_px": float(np.mean(v)) for k, v in collected.items()}
    scores["trials"] = trials
    scores["direct_p90_px"] = float(np.percentile(collected["direct"], 90))
    scores["direct_median_px"] = float(np.median(collected["direct"]))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache")
    parser.add_argument("--output-dir", default="runs/lead_time")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inputs", choices=("emg", "emg+imu"), default="emg+imu")
    parser.add_argument(
        "--model", choices=("grid", "bottleneck"), default="grid",
        help="grid = GridReachModel (train_grid_reach_model.py); bottleneck = "
        "PointingBottleneckModel (train_pointing_vae_model.py). Must match "
        "the checkpoint.",
    )
    parser.add_argument(
        "--leads-ms", type=float, nargs="+",
        default=[50, 100, 150, 200, 300, 400, 600, 800, 1000],
        help="Milliseconds before touch to predict from.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device(args.device)
    train_loader, _, test_loader = build_tracked_loaders(
        config, args.root, Path(args.cache_dir)
    )

    use_imu = args.inputs == "emg+imu"
    ablate = () if use_imu else ("imu",)
    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    builder = GridReachModel if args.model == "grid" else PointingBottleneckModel
    model = builder(config, emg_channels, imu_channels, use_imu).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()

    minimum_prefix = int(config["virtual_leader"]["minimum_prefix"])
    patch_length = int(config["model"]["patch_length"])
    rate = float(config["data"]["sample_rate_hz"]) / max(
        1, int(config["data"].get("decimation", 10))
    )
    fallback = canvas_from_disk(args.root)
    canvas_tensor = (
        torch.tensor(fallback, dtype=torch.float32, device=device) if fallback else None
    )
    mean_target = torch.stack([
        b["screen_target"].mean(0) for b in train_loader if b is not None
    ]).mean(0).cpu()

    print(f"model={args.model} inputs={args.inputs}  rate={rate:.1f} Hz")
    if fallback:
        print(f"canvas {fallback[0]:.0f} x {fallback[1]:.0f} px")
    print()
    print(f"{'lead (ms)':>10}{'trials':>8}{'direct px':>11}{'median':>9}{'p90':>8}"
          f"{'grid px':>9}{'mean px':>9}{'vs mean':>9}")
    print("-" * 73)

    history = []
    for lead_ms in sorted(args.leads_ms):
        lead_samples = max(1, int(round(lead_ms * rate / 1000.0)))
        scores = evaluate_at_lead(
            model, test_loader, lead_samples, minimum_prefix, patch_length,
            ablate, canvas_tensor, mean_target, device,
        )
        if not scores:
            print(f"{lead_ms:>10.0f}{'  no eligible trials':>30}")
            continue
        gain = (scores["mean_px"] - scores["direct_px"]) / scores["mean_px"] * 100
        print(
            f"{lead_ms:>10.0f}{scores['trials']:>8}{scores['direct_px']:>11.1f}"
            f"{scores['direct_median_px']:>9.1f}{scores['direct_p90_px']:>8.0f}"
            f"{scores['grid_px']:>9.1f}{scores['mean_px']:>9.1f}{gain:>+8.1f}%"
        )
        history.append({"lead_ms": lead_ms, **scores})

    print()
    if not history:
        return
    # Report the mean and the median separately. They answer different
    # questions and on this data they disagree: a right-skewed error
    # distribution drags the mean well above the typical trial, so quoting
    # only the mean understates how often the prediction is good, and
    # quoting only the median hides how bad the tail is.
    mean_under = [h for h in history if h["direct_px"] < 200]
    median_under = [h for h in history if h["direct_median_px"] < 200]
    best = min(history, key=lambda h: h["direct_px"])
    worst = max(history, key=lambda h: h["lead_ms"])
    slope = worst["direct_px"] - best["direct_px"]

    if mean_under:
        edge = max(mean_under, key=lambda h: h["lead_ms"])
        print(f"  MEAN under 200 px out to {edge['lead_ms']:.0f} ms before touch "
              f"({edge['direct_px']:.1f} px).")
    elif median_under:
        edge = max(median_under, key=lambda h: h["lead_ms"])
        print(f"  MEDIAN under 200 px out to {edge['lead_ms']:.0f} ms before touch "
              f"({edge['direct_median_px']:.1f} px), while the mean at that lead is "
              f"{edge['direct_px']:.1f} px.")
        print("  So the typical trial already clears 200 px and a skewed tail is")
        print("  what holds the mean above it - a tail problem, not a signal problem.")
    else:
        print(f"  Neither mean nor median reaches 200 px. Best mean is "
              f"{best['direct_px']:.1f} px at {best['lead_ms']:.0f} ms.")

    print(f"\n  Error runs {best['direct_px']:.0f} px at {best['lead_ms']:.0f} ms to "
          f"{worst['direct_px']:.0f} px at {worst['lead_ms']:.0f} ms "
          f"({slope:+.0f} px across the range).")
    if slope > 100:
        print("  Lead time is the dominant variable, so no single averaged number")
        print("  describes this model - the operating point has to be quoted with it.")
        far = [h for h in history if h["mean_px"] - h["direct_px"] < 0.05 * h["mean_px"]]
        if far:
            print(f"  Beyond ~{min(f['lead_ms'] for f in far):.0f} ms the model is at "
                  "chance: nothing about the target is\n  readable that early, which is "
                  "a statement about the reach, not the model.")

    if history:
        median_trials = float(np.median([h["trials"] for h in history]))
        print(f"\n  (median {median_trials:.0f} eligible trials per lead time; short "
              "leads keep\n   more trials, long leads drop those whose reach was "
              "shorter than the lead)")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_json({"history": history}, Path(args.output_dir) / "lead_time.json")
    print(f"\nwrote {Path(args.output_dir) / 'lead_time.json'}")


if __name__ == "__main__":
    main()
