#!/usr/bin/env python3
"""Run the complete resumable full-trajectory experiment sweep.

The sweep is deliberately mechanical: it does not choose configurations or models
from intermediate test results. Existing completed predictions are skipped, which
makes the command safe to restart after sleep, shutdown, or an interrupted process.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from emg_touch.config import load_config
from emg_touch.data.manifest import load_manifest
from emg_touch.data.schema import natural_configuration_key


MODEL_EPOCHS = {
    "imu_patch": (15, 6),
    "multimodal": (8, 5),
    "emg_tcn": (40, 10),
    "emg_patch": (40, 10),
}


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed_model(model_dir: Path) -> bool:
    return (model_dir / "best.pt").is_file() and (model_dir / "predictions.csv").is_file()


def run_command(command: list[str], label: str, failures: list[dict]) -> bool:
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
    parser.add_argument("--config", default="configs/full_trajectory.yaml")
    parser.add_argument("--artifact-root", default="artifacts/trajectory_cv")
    parser.add_argument("--run-root", default="runs/full_trajectory")
    parser.add_argument("--evaluation-root", default="evaluation/full_trajectory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["imu_patch", "multimodal", "emg_tcn", "emg_patch"],
        choices=list(MODEL_EPOCHS),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    run_root = Path(args.run_root).resolve()
    evaluation_root = Path(args.evaluation_root).resolve()
    status_path = run_root / "sweep_status.json"
    run_root.mkdir(parents=True, exist_ok=True)
    evaluation_root.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    manifest = load_manifest(config["paths"]["manifest"])
    configurations = sorted(
        manifest["configuration"].astype(str).unique(), key=natural_configuration_key
    )
    fold_count = int(config["cross_validation"]["folds"])
    python = sys.executable
    failures: list[dict] = []
    state = {
        "started": timestamp(),
        "finished": None,
        "status": "running",
        "configurations": configurations,
        "fold_count": fold_count,
        "models": args.models,
        "current": "creating folds",
        "failures": failures,
    }
    atomic_json(state, status_path)

    run_command(
        [python, "scripts/make_trajectory_folds.py", "--config", str(config_path)],
        "create deterministic folds for all configurations",
        failures,
    )

    # Prepare all fold scalers and mean baselines first. These are fast and make
    # every later model use exactly the same leakage-safe preprocessing.
    for configuration in configurations:
        for fold_index in range(fold_count):
            fold_name = f"fold-{fold_index}"
            fold_artifacts = artifact_root / configuration / fold_name
            fold_runs = run_root / configuration / fold_name
            split = fold_artifacts / "split.json"
            scaler = fold_artifacts / "scaler.npz"
            state["current"] = f"{configuration}/{fold_name}/preparation"
            atomic_json(state, status_path)
            if not split.is_file():
                failures.append({"label": str(split), "error": "missing split"})
                continue
            if not scaler.is_file():
                run_command(
                    [
                        python,
                        "scripts/fit_scaler.py",
                        "--config",
                        str(config_path),
                        "--split",
                        str(split),
                        "--output",
                        str(scaler),
                    ],
                    f"{configuration}/{fold_name}/scaler",
                    failures,
                )
            baseline_dir = fold_runs / "mean_baseline"
            if not (baseline_dir / "predictions.csv").is_file() and scaler.is_file():
                run_command(
                    [
                        python,
                        "scripts/evaluate_full_trajectory_mean.py",
                        "--config",
                        str(config_path),
                        "--split",
                        str(split),
                        "--scaler",
                        str(scaler),
                        "--output-dir",
                        str(baseline_dir),
                    ],
                    f"{configuration}/{fold_name}/mean_baseline",
                    failures,
                )

    # Run each architecture over every configuration without using intermediate
    # test metrics to decide what is trained next.
    for kind in args.models:
        epochs, patience = MODEL_EPOCHS[kind]
        for configuration in configurations:
            for fold_index in range(fold_count):
                fold_name = f"fold-{fold_index}"
                fold_artifacts = artifact_root / configuration / fold_name
                fold_runs = run_root / configuration / fold_name
                split = fold_artifacts / "split.json"
                scaler = fold_artifacts / "scaler.npz"
                model_dir = fold_runs / kind
                state["current"] = f"{configuration}/{fold_name}/{kind}"
                atomic_json(state, status_path)
                if completed_model(model_dir):
                    print(f"[{timestamp()}] SKIP completed {state['current']}", flush=True)
                    continue
                if not split.is_file() or not scaler.is_file():
                    failures.append(
                        {
                            "label": state["current"],
                            "error": "missing split or scaler",
                        }
                    )
                    continue
                command = [
                    python,
                    "scripts/train_full_trajectory.py",
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
                    str(epochs),
                    "--patience",
                    str(patience),
                ]
                if kind == "multimodal":
                    imu_checkpoint = fold_runs / "imu_patch" / "best.pt"
                    if not imu_checkpoint.is_file():
                        failures.append(
                            {
                                "label": state["current"],
                                "error": f"missing pretrained IMU checkpoint {imu_checkpoint}",
                            }
                        )
                        continue
                    command.extend(
                        [
                            "--pretrained-imu",
                            str(imu_checkpoint),
                            "--freeze-pretrained-imu",
                        ]
                    )
                run_command(command, state["current"], failures)

            prediction_files = sorted(
                (run_root / configuration).glob(f"fold-*/{kind}/predictions.csv")
            )
            if len(prediction_files) == fold_count:
                run_command(
                    [
                        python,
                        "scripts/compare_configs.py",
                        *[str(path) for path in prediction_files],
                        "--output",
                        str(evaluation_root / f"{configuration}_{kind}.csv"),
                    ],
                    f"aggregate {configuration}/{kind}",
                    failures,
                )

        all_predictions = sorted(run_root.glob(f"*/fold-*/{kind}/predictions.csv"))
        if all_predictions:
            run_command(
                [
                    python,
                    "scripts/compare_configs.py",
                    *[str(path) for path in all_predictions],
                    "--output",
                    str(evaluation_root / f"{kind}_configuration_accuracy.csv"),
                ],
                f"aggregate all configurations/{kind}",
                failures,
            )

    baseline_predictions = sorted(
        run_root.glob("*/fold-*/mean_baseline/predictions.csv")
    )
    if baseline_predictions:
        run_command(
            [
                python,
                "scripts/compare_configs.py",
                *[str(path) for path in baseline_predictions],
                "--output",
                str(evaluation_root / "mean_baseline_configuration_accuracy.csv"),
            ],
            "aggregate all configurations/mean_baseline",
            failures,
        )

    state["finished"] = timestamp()
    state["current"] = None
    state["status"] = "completed_with_failures" if failures else "completed"
    atomic_json(state, status_path)
    print(f"[{timestamp()}] SWEEP {state['status']} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
