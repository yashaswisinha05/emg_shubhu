#!/usr/bin/env python3
"""Run and summarize all causal gripper/pose architecture comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import statistics
from pathlib import Path


BASELINES = ["constant", "feature_mlp", "gru", "lstm", "tcn",
             "inceptiontime", "early_patch_transformer", "mult_cross_attention",
             "residual_gru", "residual_gru_1s",
             "frozen_classifier_residual_gru_1s"]
DISPLAY = {
    "constant": "Mean/majority", "feature_mlp": "Handcrafted + MLP",
    "gru": "Causal GRU", "lstm": "Causal LSTM", "tcn": "Causal TCN",
    "inceptiontime": "Causal InceptionTime",
    "early_patch_transformer": "Early-fusion Patch Transformer",
    "mult_cross_attention": "MulT-style cross-attention",
    "residual_gru": "State-conditioned residual GRU (ours)",
    "residual_gru_1s": "Residual GRU + 1 s intent reconstruction (ours)",
    "frozen_classifier_residual_gru_1s": (
        "Frozen-classifier residual GRU + 1 s reconstruction (ours)"),
    "proposed": "Proposed model",
}
METRICS = ["gripper_macro_f1", "position_cm", "pixel_error_px",
           "late_pixel_error_px", "future_200ms_cm", "hold_current_200ms_cm"]
REQUIRED_RUN_FILES = ("results.json", "splits.json", "emg_imu_best.pt")


def run_is_complete(directory: Path) -> bool:
    """Return true only for a run that is safe to reuse in a comparison."""
    return all((directory / name).is_file() for name in REQUIRED_RUN_FILES)


def metric_row(name, seed, directory):
    result = json.loads((directory / "results.json").read_text())["emg+imu"]
    future = result.get("future_pose_by_ms", {}).get("200", {})
    quarters = result.get("click_pixel_error_by_quarter", [None] * 4)
    checkpoint = directory / "emg_imu_best.pt"
    parameter_count = None
    if checkpoint.exists():
        import torch
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        parameter_count = state.get("parameter_count")
    return {
        "architecture": name, "model": DISPLAY[name], "seed": seed,
        "parameters": parameter_count,
        "gripper_macro_f1": result.get("gripper_macro_f1"),
        "position_cm": result.get("position_cm"),
        "pixel_error_px": result.get("click_pixel_error"),
        "late_pixel_error_px": quarters[-1],
        "future_200ms_cm": future.get("position_cm"),
        "hold_current_200ms_cm": future.get("hold_current_prediction_cm"),
    }


def write_summary(rows, output):
    fields = list(rows[0])
    with (output / "comparison_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    grouped = []
    for name in dict.fromkeys(row["architecture"] for row in rows):
        selected = [row for row in rows if row["architecture"] == name]
        summary = {"architecture": name, "model": DISPLAY[name], "runs": len(selected),
                   "parameters": selected[0]["parameters"]}
        for metric in METRICS:
            values = [row[metric] for row in selected if row[metric] is not None]
            summary[metric + "_mean"] = statistics.mean(values) if values else None
            summary[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.
        grouped.append(summary)
    summary_fields = list(grouped[0])
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader(); writer.writerows(grouped)
    lines = ["# Causal architecture comparison", "",
             "All rows use identical roots, preprocessing, trial split, targets and loss weights.", "",
             "| Model | Runs | Params | F1 ↑ | XYZ cm ↓ | Pixel px ↓ | Late pixel px ↓ | Future 200 ms cm ↓ | Hold current cm ↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def mean_std(row, metric, digits):
        mean, std = row[metric + "_mean"], row[metric + "_std"]
        if mean is None:
            return "—"
        return (f"{mean:.{digits}f}" if row["runs"] == 1
                else f"{mean:.{digits}f} ± {std:.{digits}f}")
    for row in grouped:
        params = "—" if row["parameters"] is None else f"{row['parameters']:,}"
        lines.append(
            f"| {row['model']} | {row['runs']} | {params} | "
            f"{mean_std(row, 'gripper_macro_f1', 3)} | "
            f"{mean_std(row, 'position_cm', 2)} | "
            f"{mean_std(row, 'pixel_error_px', 1)} | "
            f"{mean_std(row, 'late_pixel_error_px', 1)} | "
            f"{mean_std(row, 'future_200ms_cm', 2)} | "
            f"{mean_std(row, 'hold_current_200ms_cm', 2)} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    (output / "comparison.json").write_text(json.dumps(
        {"summary": grouped, "runs": rows}, indent=2))


def run(command, dry_run):
    print(" ".join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def run_or_resume(command, directory, *, resume, dry_run):
    if resume and run_is_complete(directory):
        print(f"RESUME: keeping completed run {directory}", flush=True)
        return
    if resume and directory.exists() and any(directory.iterdir()):
        missing = [name for name in REQUIRED_RUN_FILES
                   if not (directory / name).is_file()]
        raise RuntimeError(
            f"cannot resume incomplete run {directory}; missing {', '.join(missing)}. "
            "Move that one run directory aside, then rerun with --resume.")
    run(command, dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument(
        "--best-classifier-checkpoint",
        help="Optional causal-GRU checkpoint used only by the frozen-classifier GRU")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/gripper_pose_architecture_study"))
    parser.add_argument("--architectures", nargs="+", choices=BASELINES,
                        default=BASELINES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Fixed trial split shared by every training seed")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raw-rate-hz", type=float, default=1259.4)
    parser.add_argument("--resume", action="store_true",
                        help="Keep complete per-model runs and train only missing ones")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    classifier_checkpoint = Path(args.classifier_checkpoint)
    if not args.dry_run and not classifier_checkpoint.is_file():
        parser.error(
            f"classifier checkpoint not found: {classifier_checkpoint}. Train it first "
            "with scripts/train_gripper_neuromuscular_future.py, or pass the existing "
            "gripper_neuromuscular_future_v1 emg_imu_best.pt path. Completed baseline "
            "runs can then be retained with --resume.")
    best_classifier_checkpoint = Path(
        args.best_classifier_checkpoint or args.classifier_checkpoint)
    if not args.dry_run and not best_classifier_checkpoint.is_file():
        parser.error(f"best classifier checkpoint not found: {best_classifier_checkpoint}")
    if (not args.dry_run and not args.resume and args.output_dir.exists()
            and any(args.output_dir.iterdir())):
        parser.error("use an empty --output-dir, or pass --resume to retain complete runs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared = ["--root", *args.root, "--device", args.device,
              "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
              "--patience", str(args.patience), "--raw-rate-hz", str(args.raw_rate_hz),
              "--models", "emg+imu", "--pixel-weight", ".35",
              "--pixel-architecture", "direct", "--position-weight", "1.0",
              "--orientation-weight", "0", "--final-pose-weight", "0",
              "--future-pose-weight", ".5", "--future-pose-ms", "200"]
    python = sys.executable
    expected_splits = {}
    rows = []
    for seed in args.seeds:
        for architecture in args.architectures:
            directory = args.output_dir / architecture / f"seed{seed}"
            if architecture in {"residual_gru", "residual_gru_1s",
                                "frozen_classifier_residual_gru_1s"}:
                command = [python, "scripts/train_neuromuscular_residual_gru.py",
                           *shared, "--seed", str(seed),
                           "--split-seed", str(args.split_seed),
                           "--output-dir", str(directory)]
                if architecture in {"residual_gru_1s",
                                    "frozen_classifier_residual_gru_1s"}:
                    command.extend((
                        "--reconstruction-horizon-ms", "1000",
                        "--reconstruction-step-ms", "100",
                        "--reconstruction-decay-ms", "500",
                        "--future-imu-weight", "0"))
                if architecture == "frozen_classifier_residual_gru_1s":
                    command.extend((
                        "--classifier-checkpoint", str(best_classifier_checkpoint),
                        "--long-state-weight", "0"))
            else:
                command = [python, "scripts/train_architecture_baseline.py",
                           "--architecture", architecture, *shared,
                           "--seed", str(seed), "--split-seed", str(args.split_seed),
                           "--output-dir", str(directory)]
            run_or_resume(command, directory, resume=args.resume,
                          dry_run=args.dry_run)
        proposed = args.output_dir / "proposed" / f"seed{seed}"
        command = [python, "scripts/train_neuro_classifier_state_attention.py",
                   "--classifier-checkpoint", args.classifier_checkpoint,
                   *shared, "--seed", str(seed), "--split-seed", str(args.split_seed),
                   "--output-dir", str(proposed)]
        run_or_resume(command, proposed, resume=args.resume,
                      dry_run=args.dry_run)
        if args.dry_run:
            continue

        directories = [(name, args.output_dir / name / f"seed{seed}")
                       for name in [*args.architectures, "proposed"]]
        for name, directory in directories:
            splits = json.loads((directory / "splits.json").read_text())
            canonical = json.dumps(splits, sort_keys=True)
            expected_splits.setdefault(seed, canonical)
            if canonical != expected_splits[seed]:
                raise RuntimeError(f"split mismatch for {name}, seed {seed}")
            rows.append(metric_row(name, seed, directory))

        classifier_splits = Path(args.classifier_checkpoint).parent / "splits.json"
        if classifier_splits.exists():
            source = json.dumps(json.loads(classifier_splits.read_text()), sort_keys=True)
            if source != expected_splits[seed]:
                raise RuntimeError(
                    "the frozen classifier was trained on a different split; this would "
                    "invalidate the proposed-model comparison")
        else:
            print("WARNING: classifier splits.json is unavailable; verify that its training "
                  "set excludes every test trial before publication", file=sys.stderr)

    if not args.dry_run:
        write_summary(rows, args.output_dir)
        print(f"Comparison: {args.output_dir / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
