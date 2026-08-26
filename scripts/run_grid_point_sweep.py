#!/usr/bin/env python3
"""Run the resumable calibrated grid-and-offset configuration study."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from emg_touch.config import load_config
from emg_touch.data.manifest import load_manifest
from emg_touch.data.schema import natural_configuration_key
from emg_touch.models.grid_point import GRID_MODEL_KINDS


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def complete(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in (
            "best.pt",
            "validation_predictions.csv",
            "predictions.csv",
            "test_metrics.json",
        )
    )


def run(command: list[str], label: str, failures: list[dict[str, Any]]) -> bool:
    print(f"[{timestamp()}] START {label}", flush=True)
    print("COMMAND " + " ".join(command), flush=True)
    started = time.time()
    result = subprocess.run(command, check=False)
    elapsed = time.time() - started
    if result.returncode == 0:
        print(f"[{timestamp()}] DONE {label} elapsed_s={elapsed:.1f}", flush=True)
        return True
    failure = {
        "label": label,
        "returncode": int(result.returncode),
        "elapsed_s": elapsed,
        "time": timestamp(),
    }
    failures.append(failure)
    print(f"[{timestamp()}] FAILED {failure}", flush=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grid_point.yaml")
    parser.add_argument("--artifact-root", default="artifacts/trajectory_cv")
    parser.add_argument("--scaler-root")
    parser.add_argument("--run-root")
    parser.add_argument("--evaluation-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--configuration", action="append")
    parser.add_argument("--fold", action="append", type=int)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=GRID_MODEL_KINDS,
        default=list(GRID_MODEL_KINDS),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    manifest = load_manifest(config["paths"]["manifest"])
    if "touch_time_s" not in manifest or manifest["touch_time_s"].isna().any():
        raise ValueError(
            "The grid sweep requires touch_time_s for every trial; run "
            "scripts/build_manifest.py --config configs/grid_point.yaml"
        )
    available = sorted(
        manifest["configuration"].astype(str).unique(), key=natural_configuration_key
    )
    configurations = args.configuration or available
    unknown = set(configurations) - set(available)
    if unknown:
        raise ValueError(f"Unknown configurations {sorted(unknown)}; available={available}")
    fold_count = int(config["cross_validation"]["folds"])
    folds = args.fold or list(range(fold_count))
    if any(fold < 0 or fold >= fold_count for fold in folds):
        raise ValueError(f"Folds must be in [0, {fold_count - 1}]")

    experiment_name = str(config.get("experiment_name", "grid_point"))
    artifact_root = Path(args.artifact_root).resolve()
    scaler_root = Path(args.scaler_root or f"artifacts/{experiment_name}").resolve()
    run_root = Path(args.run_root or f"runs/{experiment_name}").resolve()
    evaluation_root = Path(
        args.evaluation_root or f"evaluation/{experiment_name}"
    ).resolve()
    for path in (scaler_root, run_root, evaluation_root):
        path.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    status_path = run_root / "sweep_status.json"
    state: dict[str, Any] = {
        "started": timestamp(),
        "finished": None,
        "status": "running",
        "configurations": configurations,
        "folds": folds,
        "models": args.models,
        "device": args.device,
        "current": None,
        "failures": failures,
    }
    atomic_json(state, status_path)
    python = sys.executable
    sweep = config.get("grid_sweep", {})
    epoch_config = sweep.get("epochs", {})
    patience_config = sweep.get("patience", {})
    required_models = set(args.models)
    if "grid_fusion" in required_models:
        required_models.add("grid_imu")
    ordered_models = [kind for kind in GRID_MODEL_KINDS if kind in required_models]

    for configuration in configurations:
        for fold in folds:
            fold_name = f"fold-{fold}"
            split = artifact_root / configuration / fold_name / "split.json"
            scaler = scaler_root / configuration / fold_name / "scaler.npz"
            fold_runs = run_root / configuration / fold_name
            if not split.is_file():
                failures.append({"label": str(split), "error": "missing split"})
                continue
            state["current"] = f"{configuration}/{fold_name}/scaler"
            atomic_json(state, status_path)
            if not scaler.is_file():
                scaler.parent.mkdir(parents=True, exist_ok=True)
                run(
                    [
                        python,
                        "scripts/fit_grid_scaler.py",
                        "--config",
                        str(config_path),
                        "--split",
                        str(split),
                        "--output",
                        str(scaler),
                    ],
                    state["current"],
                    failures,
                )
            if not scaler.is_file():
                continue

            for kind in ordered_models:
                model_dir = fold_runs / kind
                state["current"] = f"{configuration}/{fold_name}/{kind}"
                atomic_json(state, status_path)
                if complete(model_dir):
                    print(f"[{timestamp()}] SKIP completed {state['current']}", flush=True)
                    continue
                command = [
                    python,
                    "scripts/train_grid_model.py",
                    "--config",
                    str(config_path),
                    "--kind",
                    kind,
                    "--split",
                    str(split),
                    "--scaler",
                    str(scaler),
                    "--output-dir",
                    str(fold_runs),
                    "--device",
                    args.device,
                    "--epochs",
                    str(int(epoch_config.get(kind, config["training"]["epochs"]))),
                    "--patience",
                    str(int(patience_config.get(kind, config["training"]["patience"]))),
                ]
                if kind == "grid_fusion":
                    imu_checkpoint = fold_runs / "grid_imu" / "best.pt"
                    if not imu_checkpoint.is_file():
                        failures.append(
                            {"label": state["current"], "error": "missing grid_imu checkpoint"}
                        )
                        continue
                    command.extend(
                        [
                            "--pretrained-imu",
                            str(imu_checkpoint),
                            "--freeze-base-imu",
                        ]
                    )
                run(command, state["current"], failures)

    for kind in args.models:
        predictions = sorted(run_root.glob(f"*/fold-*/{kind}/predictions.csv"))
        if not predictions:
            continue
        state["current"] = f"aggregate/{kind}"
        atomic_json(state, status_path)
        run(
            [
                python,
                "scripts/compare_configs.py",
                *[str(path) for path in predictions],
                "--output",
                str(evaluation_root / f"{kind}_configuration_accuracy.csv"),
            ],
            state["current"],
            failures,
        )

    if {"grid_imu", "grid_fusion"}.issubset(required_models):
        state["current"] = "paired fusion versus IMU analysis"
        atomic_json(state, status_path)
        run(
            [
                python,
                "scripts/analyze_grid_fusion.py",
                "--run-root",
                str(run_root),
                "--output",
                str(evaluation_root / "fusion_vs_imu.csv"),
                "--seed",
                str(int(config["seed"])),
            ],
            state["current"],
            failures,
        )

    directional_predictions = [
        str(path)
        for kind in args.models
        for path in sorted(run_root.glob(f"*/fold-*/{kind}/predictions.csv"))
    ]
    if directional_predictions:
        state["current"] = "directional error analysis"
        atomic_json(state, status_path)
        run(
            [
                python,
                "scripts/analyze_directional_error.py",
                *directional_predictions,
                "--output",
                str(evaluation_root / "directional_error.csv"),
                "--seed",
                str(int(config["seed"])),
                "--grid-width",
                str(int(config["model"]["grid_size"][0])),
                "--grid-height",
                str(int(config["model"]["grid_size"][1])),
            ],
            state["current"],
            failures,
        )

        if bool(config.get("channel_attention", {}).get("enabled", False)):
            state["current"] = "channel attention analysis"
            atomic_json(state, status_path)
            run(
                [
                    python,
                    "scripts/analyze_channel_attention.py",
                    *directional_predictions,
                    "--output",
                    str(evaluation_root / "channel_attention_summary.csv"),
                ],
                state["current"],
                failures,
            )

    state["finished"] = timestamp()
    state["current"] = None
    state["status"] = "completed_with_failures" if failures else "completed"
    atomic_json(state, status_path)
    print(f"[{timestamp()}] GRID SWEEP {state['status']} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
