#!/usr/bin/env python3
"""Run one resumable study with the exact Hugging Face PatchTSTModel."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from emg_touch.config import load_config
from emg_touch.models.hf_patchtst import HF_PATCHTST_MODEL_KINDS


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
            "model_report.json",
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
    parser.add_argument("--config", default="configs/hf_patchtst_exact.yaml")
    parser.add_argument("--configuration", default="mix7")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--artifact-root", default="artifacts/trajectory_cv")
    parser.add_argument("--scaler-root")
    parser.add_argument("--run-root")
    parser.add_argument("--evaluation-root")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=HF_PATCHTST_MODEL_KINDS,
        default=list(HF_PATCHTST_MODEL_KINDS),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    experiment_name = str(config.get("experiment_name", "hf_patchtst_exact"))
    fold_name = f"fold-{args.fold}"
    artifact_root = Path(args.artifact_root).resolve()
    split = artifact_root / args.configuration / fold_name / "split.json"
    scaler_root = Path(
        args.scaler_root or f"artifacts/{experiment_name}"
    ).resolve()
    scaler = scaler_root / args.configuration / fold_name / "scaler.npz"
    run_root = Path(args.run_root or f"runs/{experiment_name}").resolve()
    evaluation_root = Path(
        args.evaluation_root or f"evaluation/{experiment_name}"
    ).resolve()
    fold_runs = run_root / args.configuration / fold_name
    for path in (scaler.parent, fold_runs, evaluation_root):
        path.mkdir(parents=True, exist_ok=True)
    if not split.is_file():
        raise FileNotFoundError(split)

    failures: list[dict[str, Any]] = []
    status_path = run_root / "study_status.json"
    state: dict[str, Any] = {
        "started": timestamp(),
        "finished": None,
        "status": "running",
        "configuration": args.configuration,
        "fold": args.fold,
        "models": args.models,
        "device": args.device,
        "current": None,
        "failures": failures,
    }
    atomic_json(state, status_path)
    python = sys.executable

    if not scaler.is_file():
        state["current"] = f"{args.configuration}/{fold_name}/scaler"
        atomic_json(state, status_path)
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
        state["finished"] = timestamp()
        state["status"] = "completed_with_failures"
        atomic_json(state, status_path)
        raise FileNotFoundError(scaler)

    study = config.get("hf_patchtst_study", {})
    epochs = study.get("epochs", {})
    patience = study.get("patience", {})
    for kind in args.models:
        model_dir = fold_runs / kind
        state["current"] = f"{args.configuration}/{fold_name}/{kind}"
        atomic_json(state, status_path)
        if complete(model_dir):
            print(f"[{timestamp()}] SKIP completed {state['current']}", flush=True)
            continue
        run(
            [
                python,
                "scripts/train_hf_patchtst.py",
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
                str(int(epochs.get(kind, config["training"]["epochs"]))),
                "--patience",
                str(int(patience.get(kind, config["training"]["patience"]))),
            ],
            state["current"],
            failures,
        )

    for kind in args.models:
        prediction = fold_runs / kind / "predictions.csv"
        if not prediction.is_file():
            continue
        state["current"] = f"aggregate/{kind}"
        atomic_json(state, status_path)
        run(
            [
                python,
                "scripts/compare_configs.py",
                str(prediction),
                "--output",
                str(evaluation_root / f"{kind}_configuration_accuracy.csv"),
            ],
            state["current"],
            failures,
        )

    if {"hf_patchtst_imu", "hf_patchtst_fusion"}.issubset(args.models):
        state["current"] = "paired fusion versus IMU analysis"
        atomic_json(state, status_path)
        run(
            [
                python,
                "scripts/analyze_model_pair.py",
                "--run-root",
                str(run_root),
                "--base-kind",
                "hf_patchtst_imu",
                "--candidate-kind",
                "hf_patchtst_fusion",
                "--output",
                str(evaluation_root / "fusion_vs_imu.csv"),
                "--seed",
                str(int(config["seed"])),
            ],
            state["current"],
            failures,
        )

    predictions = [
        str(fold_runs / kind / "predictions.csv")
        for kind in args.models
        if (fold_runs / kind / "predictions.csv").is_file()
    ]
    if predictions:
        state["current"] = "directional error analysis"
        atomic_json(state, status_path)
        run(
            [
                python,
                "scripts/analyze_directional_error.py",
                *predictions,
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

    state["finished"] = timestamp()
    state["current"] = None
    state["status"] = "completed_with_failures" if failures else "completed"
    atomic_json(state, status_path)
    print(
        f"[{timestamp()}] HF PATCHTST STUDY {state['status']} "
        f"failures={len(failures)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
