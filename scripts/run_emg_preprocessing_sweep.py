#!/usr/bin/env python3
"""Compare causal EMG filtering and sensor-local PCA on the same model.

Each cell trains the unchanged semantic-residual architecture.  Only EMG
preprocessing differs, and every seed uses its own output directory.  Existing
``results.json`` files are resumed rather than retrained.

Example:

    python scripts/run_emg_preprocessing_sweep.py \
      --root "/media/.../emg_imu_vive" \
      --config configs/tracked_semantic_residual_distillation.yaml \
      --cache-dir artifacts/tracked_cache_posture \
      --output-dir runs/emg_preprocessing_sweep \
      --device cuda --teacher-epochs 25 --epochs 50 --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "bandpass": {
        "emg_bandpass_hz": [20.0, 450.0],
        "emg_filter_order": 4,
    },
    "bandpass_notch": {
        "emg_bandpass_hz": [20.0, 450.0],
        "emg_filter_order": 4,
        "emg_notch_hz": 50.0,
        "emg_notch_quality": 30.0,
    },
    "bandpass_pca": {
        "emg_bandpass_hz": [20.0, 450.0],
        "emg_filter_order": 4,
        "emg_pca_components_per_sensor": 8,
    },
}


def _clean_preprocessing(config: dict[str, Any]) -> None:
    for key in (
        "emg_bandpass_hz", "emg_filter_order", "emg_notch_hz",
        "emg_notch_quality", "emg_pca_components_per_sensor",
        "emg_pca_means", "emg_pca_components",
        "emg_pca_explained_variance_ratio",
    ):
        config["data"].pop(key, None)


def _mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f}±{statistics.stdev(values):.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--config",
        default="configs/tracked_semantic_residual_distillation.yaml",
    )
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache_posture")
    parser.add_argument("--output-dir", default="runs/emg_preprocessing_sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-epochs", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--lead-window-ms", type=float, nargs=2, default=[50, 400])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        template = yaml.safe_load(handle)
    root_output = Path(args.output_dir)
    config_output = root_output / "configs"
    config_output.mkdir(parents=True, exist_ok=True)
    trainer = (
        Path(__file__).resolve().parent
        / "train_semantic_residual_distillation_model.py"
    )
    collected: dict[str, list[dict[str, float]]] = {}
    failures: list[str] = []

    for variant in args.variants:
        for seed in args.seeds:
            run_name = f"{variant}__seed{seed}"
            run_output = root_output / run_name
            results_path = run_output / "results.json"
            if results_path.exists():
                print(f"[skip] {run_name}: results.json already exists")
            else:
                config = json.loads(json.dumps(template))
                _clean_preprocessing(config)
                config["data"].update(VARIANTS[variant])
                config["seed"] = int(seed)
                generated_config = config_output / f"{run_name}.yaml"
                with generated_config.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(config, handle, sort_keys=False)
                command = [
                    sys.executable,
                    str(trainer),
                    "--root", args.root,
                    "--config", str(generated_config),
                    "--cache-dir", args.cache_dir,
                    "--output-dir", str(run_output),
                    "--device", args.device,
                    "--seed", str(seed),
                    "--teacher-epochs", str(args.teacher_epochs),
                    "--epochs", str(args.epochs),
                    "--finetune-epochs", str(args.finetune_epochs),
                    "--lead-window-ms", *map(str, args.lead_window_ms),
                ]
                print(f"\n=== {run_name} ===", flush=True)
                completed = subprocess.run(command)
                if completed.returncode:
                    failures.append(run_name)
                    print(f"FAILED: {run_name} (exit {completed.returncode})")
                    continue
            if not results_path.exists():
                failures.append(run_name)
                continue
            with results_path.open("r", encoding="utf-8") as handle:
                test = json.load(handle).get("test", {})
            collected.setdefault(variant, []).append(test)

    if not collected:
        raise SystemExit("no preprocessing run produced results")
    print("\n=== held-out preprocessing comparison (mean±SD across seeds) ===")
    print(
        f"{'variant':19} {'student px':>13} {'EMG-only':>13} "
        f"{'remove EMG':>13} {'shuffle EMG':>13} {'residual':>11}"
    )
    print("-" * 88)
    rows = []
    for variant, runs in collected.items():
        student = [float(run["student_px"]) for run in runs]
        emg_only = [float(run["emg_only_px"]) for run in runs]
        remove = [
            float(run["without_emg_px"] - run["student_px"]) for run in runs
        ]
        shuffled = [
            float(run["shuffled_emg_px"] - run["student_px"])
            for run in runs if "shuffled_emg_px" in run
        ]
        residual = [float(run.get("residual_gain_px", 0.0)) for run in runs]
        rows.append((statistics.mean(student), variant))
        print(
            f"{variant:19} {_mean_std(student):>13} {_mean_std(emg_only):>13} "
            f"{_mean_std(remove):>13} {_mean_std(shuffled):>13} "
            f"{_mean_std(residual):>11}"
        )
    winner = min(rows)[1]
    print(
        f"\nlowest mean student error: {winner}. Confirm it also has positive "
        "remove/shuffle EMG contributions before adopting it."
    )
    summary = {
        "variants": collected,
        "winner_by_student_px": winner,
        "failures": failures,
    }
    with (root_output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {root_output / 'summary.json'}")


if __name__ == "__main__":
    main()
