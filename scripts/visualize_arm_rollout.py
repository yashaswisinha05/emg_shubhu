#!/usr/bin/env python3
"""Render a standalone, self-contained HTML viewer of the Hill physics
branch's joint-angle rollout on real trials.

Produces one .html file with no external dependencies (no CDN, no network)
that opens directly in a browser on any machine - the checkpoint does not
need to travel with it, only the exported trajectories do. Two views: manual
slider control of the two joint angles (a data-independent kinematics sanity
check), and playback of real rollouts with the click target and both the
physics-only and blended predictions overlaid.

Usage:
  python scripts/visualize_arm_rollout.py \
    --config configs/hill_fusion.yaml \
    --checkpoint runs/hill_fusion/a1/fold-0/grid_fusion_physics/best.pt \
    --split artifacts/trajectory_cv/a1/fold-0/split.json \
    --scaler artifacts/hill_fusion/a1/fold-0/scaler.npz \
    --output evaluation/hill_fusion/arm_rollout.html

Without --checkpoint, the model is used at random initialization - useful to
confirm the pipeline and rendering are correct before any training exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from emg_touch.config import load_config
from emg_touch.data.grid_trajectory import build_grid_trajectory_loaders
from emg_touch.models.grid_point import build_grid_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hill_fusion.yaml")
    parser.add_argument("--kind", default="grid_fusion_physics")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split")
    parser.add_argument("--scaler")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument(
        "--output", default="evaluation/hill_fusion/arm_rollout.html"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model = build_grid_model(args.kind, config)
    checkpoint_label = "random initialization (no --checkpoint given)"
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        checkpoint_label = (
            f"{args.checkpoint} (epoch {checkpoint.get('epoch', '?')})"
        )
    model.eval()

    _, val_loader, _ = build_grid_trajectory_loaders(
        config,
        split_path=args.split or config["paths"]["split_file"],
        scaler_path=args.scaler or config["paths"]["scaler"],
    )
    batch = next(iter(val_loader))
    with torch.no_grad():
        outputs = model(batch)

    if "physics_trajectory" not in outputs:
        raise ValueError(
            f"{args.kind} has no physics branch; use grid_fusion_physics"
        )

    decimation = int(config.get("physics", {}).get("decimation", 4))
    trajectory = outputs["physics_trajectory"]
    lengths = batch["lengths"]
    count = min(args.trials, trajectory.size(0))

    trials = []
    for index in range(count):
        steps = int((lengths[index].item() + decimation - 1) // decimation)
        trial_id = batch["trial_id"][index]
        trials.append(
            {
                "id": str(trial_id).split("__")[-1],
                "theta": [
                    [round(a, 4), round(b, 4)]
                    for a, b in trajectory[index, :steps].tolist()
                ],
                "target": [round(v, 4) for v in batch["target"][index].tolist()],
                "physics_pred": [
                    round(v, 4)
                    for v in outputs["physics_prediction"][index].tolist()
                ],
                "fusion_pred": [
                    round(v, 4)
                    for v in outputs["fusion_prediction"][index].tolist()
                ],
                "blend": round(float(outputs["physics_blend"][index]), 5),
            }
        )

    template_path = Path(__file__).parent / "_arm_rollout_template.html"
    html = template_path.read_text()
    html = html.replace("__TRIALS_JSON__", json.dumps(trials))
    html = html.replace("__CHECKPOINT_LABEL__", checkpoint_label)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Wrote {output_path.resolve()}")
    print(f"Checkpoint: {checkpoint_label}")
    print(f"Trials: {count}   Decimation: {decimation}")
    mean_blend = sum(t["blend"] for t in trials) / len(trials)
    print(f"Mean physics_blend across these trials: {mean_blend:.5f}")


if __name__ == "__main__":
    main()
