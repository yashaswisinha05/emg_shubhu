#!/usr/bin/env python3
"""TimeSiam-inspired pre-training and fine-tuning for 200 ms pose intent."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.models.reach_grasp_timesiam_future import TimeSiamFutureGripperPoseModel
from emg_touch.models.timesiam_wearable_loss import future_wearable_loss
from scripts import train_gripper_state_pose as base


def selection_score(report, args):
    score = (report["position_cm"] + .1 * (report["orientation_deg"] or 0)
             + 5 * (1 - report["gripper_macro_f1"]))
    quarters = report.get("click_pixel_error_by_quarter", [None] * 4)
    late = quarters[-1] if quarters[-1] is not None else report.get("click_pixel_error", 0.)
    score += .005 * report.get("click_pixel_error", 0.) + .03 * late
    score += report.get("final_position_cm", 0.)
    score += .1 * (report.get("final_orientation_deg") or 0.)
    return score


def validation_pretrain_loss(model, trials, stats, args, modality):
    model.eval()
    values = []
    with torch.no_grad():
        for batch in base.batches(trials, stats, args.batch_size, args.device):
            output = model(batch["emg"], batch["imu"])
            values.append(future_wearable_loss(output, batch, modality).item())
    return float(np.mean(values))


def train_one(modality, args, train, validation, stats, class_weight, settings,
              augmenter=None):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model_args = {
        "modality": modality, "width": EXTRA.width, "patch": EXTRA.patch,
        "stride": EXTRA.stride, "layers": EXTRA.layers, "heads": EXTRA.heads,
        "dropout": .1, "react_context": 100,
        "predict_click": args.pixel_weight > 0,
        "predict_final_pose": args.final_pose_weight > 0,
        "future_steps": args.future_pose_ms // 10,
        "lineage_context": EXTRA.lineage_context,
    }
    model = TimeSiamFutureGripperPoseModel(**model_args).to(args.device)
    suffix = modality.replace("+", "_")
    pretrain_path = args.output_dir / f"{suffix}_timesiam_pretrained.pt"

    optimizer = torch.optim.AdamW(model.parameters(), lr=EXTRA.pretrain_lr, weight_decay=1e-4)
    best_pretrain, best_state = float("inf"), None
    pretrain_history = []
    for epoch in range(1, EXTRA.pretrain_epochs + 1):
        model.train(); losses = []
        for batch in base.batches(train, stats, args.batch_size, args.device, True):
            if augmenter is not None:
                batch = augmenter(batch, stats, modality)
            optimizer.zero_grad(set_to_none=True)
            value = future_wearable_loss(model(batch["emg"], batch["imu"]), batch, modality)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); losses.append(value.item())
        validation_loss = validation_pretrain_loss(model, validation, stats, args, modality)
        pretrain_history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                                 "validation_loss": validation_loss})
        print(f"timesiam_pretrain modality={modality} epoch={epoch} "
              f"loss={np.mean(losses):.4f} val={validation_loss:.4f}", flush=True)
        if validation_loss < best_pretrain:
            best_pretrain = validation_loss
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"format": "gripper_timesiam_pretrain_v1", "state_dict": best_state,
                "model_args": model_args, "normalization": stats,
                "preprocessing": settings, "validation_loss": best_pretrain}, pretrain_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    checkpoint = args.output_dir / f"{suffix}_best.pt"
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in base.batches(train, stats, args.batch_size, args.device, True):
            if augmenter is not None:
                batch = augmenter(batch, stats, modality)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["emg"], batch["imu"])
            value = base.loss(model, output, batch, class_weight, args)
            value = value + EXTRA.reconstruction_weight * future_wearable_loss(
                output, batch, modality)
            value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); losses.append(value.item())
        report = base.evaluate(model, validation, stats, args)
        score = selection_score(report, args)
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "selection_score": score, "validation": report})
        print(f"timesiam_finetune modality={modality} epoch={epoch} loss={np.mean(losses):.4f} "
              f"score={score:.3f} state_f1={report['gripper_macro_f1']:.3f} "
              f"pose={report['position_cm']:.2f}cm click={report['click_pixel_error']:.1f}px "
              f"endpoint={report['final_position_cm']:.2f}cm", flush=True)
        if score < best:
            best, stale = score, 0
            torch.save({"format": "gripper_timesiam_future_v1",
                        "state_dict": model.state_dict(), "model_args": model_args,
                        "normalization": stats, "preprocessing": settings,
                        "classes": ["open", "close"], "validation": report,
                        "seed": args.seed,
                        "timesiam": {"pretrain_epochs": EXTRA.pretrain_epochs,
                                     "lineage_context": EXTRA.lineage_context,
                                     "reconstruction_weight": EXTRA.reconstruction_weight,
                                     "future_ms": args.future_pose_ms}}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    (args.output_dir / f"{suffix}_timesiam_history.json").write_text(json.dumps(
        {"pretrain": pretrain_history, "finetune": history}, indent=2))
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    return model, state


def main():
    global EXTRA
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--timesiam-pretrain-epochs", type=int, default=15)
    parser.add_argument("--timesiam-pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--timesiam-reconstruction-weight", type=float, default=.05)
    parser.add_argument("--timesiam-lineage-context", type=int, default=25)
    parser.add_argument("--timesiam-width", type=int, default=128)
    parser.add_argument("--timesiam-patch", type=int, default=16)
    parser.add_argument("--timesiam-stride", type=int, default=4)
    parser.add_argument("--timesiam-layers", type=int, default=4)
    parser.add_argument("--timesiam-heads", type=int, default=4)
    EXTRA, remaining = parser.parse_known_args()
    EXTRA.pretrain_epochs = EXTRA.timesiam_pretrain_epochs
    EXTRA.pretrain_lr = EXTRA.timesiam_pretrain_lr
    EXTRA.reconstruction_weight = EXTRA.timesiam_reconstruction_weight
    EXTRA.lineage_context = EXTRA.timesiam_lineage_context
    EXTRA.width = EXTRA.timesiam_width
    EXTRA.patch = EXTRA.timesiam_patch
    EXTRA.stride = EXTRA.timesiam_stride
    EXTRA.layers = EXTRA.timesiam_layers
    EXTRA.heads = EXTRA.timesiam_heads
    if min(EXTRA.pretrain_epochs, EXTRA.lineage_context) <= 0:
        parser.error("pretrain epochs and lineage context must be positive")
    if EXTRA.pretrain_lr <= 0:
        parser.error("--timesiam-pretrain-lr must be positive")
    if EXTRA.reconstruction_weight < 0:
        parser.error("--timesiam-reconstruction-weight cannot be negative")
    if min(EXTRA.width, EXTRA.patch, EXTRA.stride, EXTRA.layers, EXTRA.heads) <= 0:
        parser.error("TimeSiam model dimensions must be positive")
    if EXTRA.width % EXTRA.heads:
        parser.error("--timesiam-width must be divisible by --timesiam-heads")
    sys.argv = [sys.argv[0], *remaining]
    required = {
        "--pixel-architecture": "goal-consistent", "--pixel-weight": "0.35",
        "--final-pose-weight": "0.2", "--endpoint-loss-space": "physical",
        "--final-position-multiplier": "2.5", "--endpoint-progress-weight": "1.0",
        "--future-pose-ms": "200", "--future-pose-weight": "0.1",
    }
    present = set(remaining)
    for option, value in required.items():
        if option not in present:
            sys.argv.extend([option, value])
    # Patch only for this entry point and restore it afterwards. This keeps an
    # in-process caller (including the test suite) from changing the legacy
    # trainer's behaviour after this experiment returns.
    original_train_one = base.train_one
    base.train_one = train_one
    try:
        base.main()
    finally:
        base.train_one = original_train_one
    output = (Path(sys.argv[sys.argv.index("--output-dir") + 1])
              if "--output-dir" in sys.argv else Path("runs/gripper_state_pose_seed42"))
    results_path = output / "results.json"
    results = json.loads(results_path.read_text())
    results["protocol"]["future_architecture"] = "TimeSiam-inspired lineage cross-attention"
    results["protocol"]["timesiam_pretrain_epochs"] = EXTRA.pretrain_epochs
    results_path.write_text(json.dumps(results, indent=2))
    print(f"TimeSiam protocol written to {results_path}")


if __name__ == "__main__":
    main()
