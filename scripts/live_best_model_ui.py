#!/usr/bin/env python3
"""Run the existing red-target/cyan-prediction UI with a distilled best model.

This has the same trial playback and visual layout as ``live_prediction_ui``.
It loads channel+horizon or semantic-residual checkpoints, applies their exact
training normalization/filter/PCA pipeline, and calls only
``student_forward(EMG, IMU, time_mask)``.  Recorded VIVE is used only to place
the red ground-truth target and calculate the displayed pixel error.

    python scripts/live_best_model_ui.py \
      --trial-root "/media/.../any_compatible_tracked_dataset" \
      --sweep-dir runs/emg_preprocessing_sweep \
      --device cuda --speed 1.0 --prediction-delay-ms 600

``--sweep-dir`` selects the preprocessing variant by mean validation error
across completed seeds and then loads that variant's best-validation seed.  It
never inspects test error for model selection.  By default all recursively
discovered trial folders under ``--trial-root`` are available in the UI.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts.live_prediction_ui import PredictionReplayWindow  # noqa: E402
from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    apply_sensor_local_pca,
    discover_trials,
    preprocess_tracked_trial,
    session_emg_scale,
    session_imu_statistics,
)
from emg_touch.live_distillation import (  # noqa: E402
    LiveDistillationModel,
    preprocessing_signature,
)

from PySide6.QtWidgets import QApplication  # noqa: E402


def select_best_sweep_checkpoint(
    sweep_directory: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Choose a complete variant by mean validation, then its best-val seed."""
    root = Path(sweep_directory)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for results_path in sorted(root.glob("*__seed*/results.json")):
        run_name = results_path.parent.name
        if "__seed" not in run_name:
            continue
        variant, raw_seed = run_name.rsplit("__seed", 1)
        try:
            seed = int(raw_seed)
            with results_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        validation = [
            float(record["student_px"])
            for record in payload.get("history", [])
            if record.get("phase") in {"student", "finetune"}
            and "student_px" in record
        ]
        checkpoint = results_path.parent / "best.pt"
        if not validation or not checkpoint.is_file():
            continue
        grouped.setdefault(variant, []).append({
            "seed": seed,
            "validation_px": min(validation),
            "checkpoint": checkpoint,
        })
    if not grouped:
        raise FileNotFoundError(
            f"no completed *__seed*/results.json + best.pt runs under {root}"
        )
    maximum_runs = max(len(runs) for runs in grouped.values())
    complete = {
        variant: runs
        for variant, runs in grouped.items()
        if len(runs) == maximum_runs
    }
    variant_scores = {
        variant: statistics.mean(run["validation_px"] for run in runs)
        for variant, runs in complete.items()
    }
    winner = min(variant_scores, key=variant_scores.get)
    selected = min(
        complete[winner], key=lambda run: (run["validation_px"], run["seed"])
    )
    details = {
        "variant": winner,
        "seed": selected["seed"],
        "run_validation_px": selected["validation_px"],
        "variant_mean_validation_px": variant_scores[winner],
        "runs_per_complete_variant": maximum_runs,
        "variant_scores": variant_scores,
    }
    return Path(selected["checkpoint"]), details


def _session_owner(
    path: Path, prefixes: list[str], fallback: str
) -> str | None:
    if not prefixes:
        return fallback
    for part in reversed(path.parts):
        lowered = part.lower()
        for raw_prefix in prefixes:
            prefix = str(raw_prefix).strip().lower()
            if (
                lowered == prefix
                or lowered.startswith(prefix + "_")
                or lowered.startswith(prefix + "-")
            ):
                return part
    return None


class StudentForwardAdapter(nn.Module):
    """Expose the wearable student through the legacy UI's forward contract."""

    def __init__(self, distilled_model: nn.Module, context_samples: int) -> None:
        super().__init__()
        self.distilled_model = distilled_model
        self.context_samples = int(context_samples)

    def forward(
        self,
        emg: torch.Tensor,
        imu: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if emg.size(1) < self.context_samples:
            missing = self.context_samples - emg.size(1)
            emg = torch.nn.functional.pad(emg, (0, 0, missing, 0))
            imu = torch.nn.functional.pad(imu, (0, 0, missing, 0))
            time_mask = torch.nn.functional.pad(
                time_mask, (missing, 0), value=False
            )
        return self.distilled_model.student_forward(
            emg, imu, time_mask, sample=False
        )


class NormalizedReplayTrialLoader:
    """Reproduce training preprocessing for one recorded sensor placement."""

    def __init__(
        self,
        config: dict[str, Any],
        session_trials: dict[str, list[Path]],
        trial_sessions: dict[str, str],
    ) -> None:
        self.config = config
        self.data_config = config["data"]
        self.session_trials = session_trials
        self.trial_sessions = trial_sessions
        self.normalization: dict[
            str, tuple[Any, tuple[Any, Any]]
        ] = {}

    def _statistics(self, session: str) -> tuple[Any, tuple[Any, Any]]:
        if session not in self.normalization:
            trials = self.session_trials[session]
            emg_scale = session_emg_scale(trials, self.data_config)
            imu_statistics = session_imu_statistics(trials, self.data_config)
            if emg_scale is None or imu_statistics is None:
                raise RuntimeError(
                    f"could not calculate replay normalization for {session}"
                )
            self.normalization[session] = (emg_scale, imu_statistics)
        return self.normalization[session]

    def __call__(self, path: Path) -> dict[str, Any] | None:
        data = preprocess_tracked_trial(path, self.data_config)
        if data is None:
            return None
        session = self.trial_sessions[str(path)]
        emg_scale, (imu_center, imu_scale) = self._statistics(session)
        data["emg"] = apply_sensor_local_pca(
            data["emg"] / emg_scale, self.data_config
        )
        data["imu"] = (data["imu"] - imu_center) / imu_scale
        return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", required=True)
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--checkpoint")
    model_source.add_argument(
        "--sweep-dir",
        help="Automatically select the validation-best completed sweep model",
    )
    parser.add_argument(
        "--config",
        help="Optional compatibility check; checkpoint config remains authoritative",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--prediction-delay-ms",
        type=float,
        default=600.0,
        help="Wait this long after movement onset before the first prediction",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--session-prefixes",
        nargs="+",
        help="Optional inference-dataset folder filter; default uses all sessions",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    selection = None
    if args.sweep_dir:
        checkpoint, selection = select_best_sweep_checkpoint(args.sweep_dir)
        print(
            f"validation-selected sweep model: {selection['variant']} seed "
            f"{selection['seed']} | run={selection['run_validation_px']:.1f}px | "
            f"variant mean={selection['variant_mean_validation_px']:.1f}px"
        )
    assert checkpoint is not None
    runner = LiveDistillationModel("Best wearable model", checkpoint, args.device)
    config = runner.config
    if args.config:
        external = load_config(args.config)
        if preprocessing_signature(external) != preprocessing_signature(config):
            raise SystemExit(
                "--config preprocessing does not match the checkpoint. Use the "
                "configuration that trained this checkpoint; the embedded fitted "
                "PCA, when present, is loaded from the checkpoint itself."
            )

    discovered = discover_trials(args.trial_root)
    prefixes = list(args.session_prefixes or [])
    session_trials: dict[str, list[Path]] = {}
    trial_sessions: dict[str, str] = {}
    for name, paths in discovered.items():
        for path in paths:
            owner = _session_owner(path, prefixes, name)
            if owner is not None:
                session_trials.setdefault(owner, []).append(path)
                trial_sessions[str(path)] = owner
    trials = [path for paths in session_trials.values() for path in paths]
    if not trials:
        raise SystemExit(
            f"no matching trial_*.csv found under {args.trial_root}"
        )
    if args.shuffle:
        random.shuffle(trials)

    trial_loader = NormalizedReplayTrialLoader(
        config, session_trials, trial_sessions
    )
    model = StudentForwardAdapter(
        runner.model, runner.context_samples
    ).to(runner.device).eval()
    print(
        f"loaded {checkpoint} ({runner.kind}) | {len(trials)} trial(s) | "
        f"device={runner.device}"
    )
    print(
        "deployment check: predictions receive causal EMG+IMU only; VIVE/target "
        "are display-only ground truth"
    )
    print(
        f"first prediction: {args.prediction_delay_ms:.0f} ms after detected "
        "movement onset"
    )

    app = QApplication(sys.argv)
    window = PredictionReplayWindow(
        trials,
        model,
        config,
        runner.device,
        speed=args.speed,
        trial_loader=trial_loader,
        window_title="Best Wearable Model — Live Prediction Replay",
        maximum_prefix=runner.context_samples,
        prediction_delay_ms=args.prediction_delay_ms,
    )
    window.resize(1100, 720)
    window.show()
    window.start_next_trial()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
