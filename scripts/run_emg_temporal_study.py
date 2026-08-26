#!/usr/bin/env python3
"""Run a resumable, touch-aligned EMG temporal contribution study.

For every selected configuration and fold, this script trains one causal IMU
baseline and then compares EMG-only and controlled EMG-residual models across predeclared EMG
windows.  All conditions use the same split and the same causal training-fold
scaler.  Existing complete outputs are skipped on restart.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from emg_touch.config import load_config, save_config
from emg_touch.data.manifest import load_manifest
from emg_touch.data.schema import natural_configuration_key


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed_model(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in ("best.pt", "validation_predictions.csv", "predictions.csv")
    )


def run_command(command: list[str], label: str, failures: list[dict[str, Any]]) -> bool:
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


def study_windows(config: dict[str, Any]) -> list[dict[str, Any]]:
    windows = config.get("temporal_study", {}).get("windows", [])
    if not windows:
        raise ValueError("temporal_study.windows is empty")
    result = []
    seen = set()
    for item in windows:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError(f"Invalid temporal window: {item!r}")
        name = str(item["name"])
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
            raise ValueError(f"Unsafe temporal window name: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate temporal window name: {name}")
        seen.add(name)
        start = item.get("start_s")
        end = item.get("end_s")
        start = None if start is None else float(start)
        end = None if end is None else float(end)
        if end is not None and end > 0.0:
            raise ValueError(f"Post-touch window is forbidden: {item}")
        if start is not None and end is not None and start > end:
            raise ValueError(f"Window start exceeds end: {item}")
        result.append({"name": name, "start_s": start, "end_s": end})
    if "causal_all" not in seen:
        raise ValueError("temporal_study.windows must include a causal_all window")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/emg_temporal_study.yaml")
    parser.add_argument("--artifact-root", default="artifacts/trajectory_cv")
    parser.add_argument("--scaler-root", default="artifacts/emg_temporal_touch")
    parser.add_argument("--run-root", default="runs/emg_temporal_touch")
    parser.add_argument("--evaluation-root", default="evaluation/emg_temporal_touch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--configuration",
        action="append",
        help="Configuration to run; repeat as needed. Omit for all configurations.",
    )
    parser.add_argument(
        "--fold",
        action="append",
        type=int,
        help="Fold to run; repeat as needed. Omit for all folds.",
    )
    parser.add_argument(
        "--window",
        action="append",
        help="Window name to run; repeat as needed. Omit for all declared windows.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["emg_patch", "emg_residual"],
        default=["emg_patch", "emg_residual"],
        help="Window-specific models. A causal IMU baseline is always trained.",
    )
    args = parser.parse_args()
    if "emg_residual" not in args.models:
        raise ValueError(
            "--models must include emg_residual because incremental EMG value is "
            "defined relative to the causal IMU model"
        )

    base_config = load_config(args.config)
    if str(base_config["data"].get("trajectory_end", "")).lower() != "touch_time":
        raise ValueError("The temporal study requires data.trajectory_end=touch_time")
    if str(base_config["data"].get("temporal_anchor", "")).lower() != "touch_time":
        raise ValueError("The temporal study requires data.temporal_anchor=touch_time")
    windows = study_windows(base_config)
    available_window_names = {item["name"] for item in windows}
    selected_window_names = set(args.window or available_window_names)
    unknown_windows = selected_window_names - available_window_names
    if unknown_windows:
        raise ValueError(
            f"Unknown windows {sorted(unknown_windows)}; "
            f"available={sorted(available_window_names)}"
        )
    selected_windows = [
        item for item in windows if item["name"] in selected_window_names
    ]

    manifest = load_manifest(base_config["paths"]["manifest"])
    available_configs = sorted(
        manifest["configuration"].astype(str).unique(), key=natural_configuration_key
    )
    configurations = args.configuration or available_configs
    unknown_configs = set(configurations) - set(available_configs)
    if unknown_configs:
        raise ValueError(
            f"Unknown configurations {sorted(unknown_configs)}; available={available_configs}"
        )
    fold_count = int(base_config["cross_validation"]["folds"])
    folds = args.fold or list(range(fold_count))
    if any(fold < 0 or fold >= fold_count for fold in folds):
        raise ValueError(f"Folds must be in [0, {fold_count - 1}]")

    artifact_root = Path(args.artifact_root).resolve()
    scaler_root = Path(args.scaler_root).resolve()
    run_root = Path(args.run_root).resolve()
    evaluation_root = Path(args.evaluation_root).resolve()
    config_root = run_root / "_window_configs"
    status_path = run_root / "study_status.json"
    for path in (scaler_root, run_root, evaluation_root, config_root):
        path.mkdir(parents=True, exist_ok=True)

    window_configs: dict[str, Path] = {}
    for window in windows:
        config = deepcopy(base_config)
        config["data"]["trajectory_end"] = "touch_time"
        config["data"]["temporal_anchor"] = "touch_time"
        config["data"]["temporal_label"] = window["name"]
        config["data"]["emg_window_s"] = [window["start_s"], window["end_s"]]
        output = config_root / f"{window['name']}.yaml"
        save_config(config, output)
        window_configs[window["name"]] = output

    study = base_config.get("temporal_study", {})
    epoch_config = study.get("epochs", {})
    patience_config = study.get("patience", {})
    python = sys.executable
    failures: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "started": timestamp(),
        "finished": None,
        "status": "running",
        "configurations": configurations,
        "folds": folds,
        "windows": [item["name"] for item in selected_windows],
        "models": args.models,
        "device": args.device,
        "current": None,
        "failures": failures,
    }
    atomic_json(state, status_path)

    causal_config = window_configs["causal_all"]
    for configuration in configurations:
        for fold in folds:
            fold_name = f"fold-{fold}"
            split = artifact_root / configuration / fold_name / "split.json"
            scaler = scaler_root / configuration / fold_name / "scaler.npz"
            fold_runs = run_root / configuration / fold_name
            if not split.is_file():
                failures.append({"label": str(split), "error": "missing split"})
                continue

            state["current"] = f"{configuration}/{fold_name}/causal_scaler"
            atomic_json(state, status_path)
            if not scaler.is_file():
                scaler.parent.mkdir(parents=True, exist_ok=True)
                run_command(
                    [
                        python,
                        "scripts/fit_scaler.py",
                        "--config",
                        str(causal_config),
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

            imu_dir = fold_runs / "causal_all" / "imu_patch"
            state["current"] = f"{configuration}/{fold_name}/causal_all/imu_patch"
            atomic_json(state, status_path)
            if not completed_model(imu_dir):
                run_command(
                    [
                        python,
                        "scripts/train_full_trajectory.py",
                        "--config",
                        str(causal_config),
                        "--kind",
                        "imu_patch",
                        "--split",
                        str(split),
                        "--scaler",
                        str(scaler),
                        "--output-dir",
                        str(fold_runs / "causal_all"),
                        "--device",
                        args.device,
                        "--epochs",
                        str(int(epoch_config.get("imu_patch", 15))),
                        "--patience",
                        str(int(patience_config.get("imu_patch", 6))),
                    ],
                    state["current"],
                    failures,
                )
            if not completed_model(imu_dir):
                continue

            for window in selected_windows:
                window_name = window["name"]
                window_root = fold_runs / window_name
                config_path = window_configs[window_name]
                for kind in args.models:
                    model_dir = window_root / kind
                    state["current"] = (
                        f"{configuration}/{fold_name}/{window_name}/{kind}"
                    )
                    atomic_json(state, status_path)
                    if completed_model(model_dir):
                        print(f"[{timestamp()}] SKIP completed {state['current']}", flush=True)
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
                        str(window_root),
                        "--device",
                        args.device,
                        "--epochs",
                        str(int(epoch_config.get(kind, 40))),
                        "--patience",
                        str(int(patience_config.get(kind, 10))),
                    ]
                    if kind == "emg_residual":
                        command.extend(
                            [
                                "--pretrained-imu",
                                str(imu_dir / "best.pt"),
                                "--freeze-pretrained-imu",
                            ]
                        )
                    run_command(command, state["current"], failures)

    state["current"] = "paired temporal analysis"
    atomic_json(state, status_path)
    analysis_command = [
        python,
        "scripts/analyze_emg_temporal_study.py",
        "--run-root",
        str(run_root),
        "--output-dir",
        str(evaluation_root),
        "--bootstrap-repeats",
        str(int(study.get("bootstrap_repeats", 5000))),
        "--seed",
        str(int(base_config["seed"])),
    ]
    for window in selected_windows:
        analysis_command.extend(["--window", window["name"]])
    run_command(
        analysis_command,
        "paired temporal analysis",
        failures,
    )

    state["finished"] = timestamp()
    state["current"] = None
    state["status"] = "completed_with_failures" if failures else "completed"
    atomic_json(state, status_path)
    print(f"[{timestamp()}] STUDY {state['status']} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
