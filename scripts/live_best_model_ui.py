#!/usr/bin/env python3
"""Run the existing red-target/cyan-prediction UI with a distilled best model.

This has the same trial playback and visual layout as ``live_prediction_ui``.
It loads channel+horizon or semantic-residual checkpoints, applies their exact
training normalization/filter/PCA pipeline, and calls only
``student_forward(EMG, IMU, time_mask)``.  Recorded VIVE is used only to place
the red ground-truth target and calculate the displayed pixel error.

    python scripts/live_best_model_ui.py \
      --trial-root "/media/.../emg_imu_vive" \
      --checkpoint runs/semantic_residual_distillation/best.pt \
      --config configs/tracked_semantic_residual_distillation.yaml \
      --device cuda --speed 1.0 --prediction-delay-ms 600
"""
from __future__ import annotations

import argparse
import random
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
    parser.add_argument("--checkpoint", required=True)
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
    args = parser.parse_args()

    runner = LiveDistillationModel("Best wearable model", args.checkpoint, args.device)
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
    prefixes = list(config["data"].get("include_session_prefixes", []))
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
        f"loaded {args.checkpoint} ({runner.kind}) | {len(trials)} trial(s) | "
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
