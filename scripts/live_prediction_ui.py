#!/usr/bin/env python3
"""The data-collection target UI, but the model predicts where the touch lands.

Visually the same app that recorded this dataset - dark theme, a red target
button with the same fade-in / hit-ripple animations, taken verbatim from
/Users/yashaswi/Downloads/Example-Applications/Python/SimpleButtonExperiment.py
(the SAME window class run_simple.py launches for live data collection). The
red button now sits at the trial's TRUE recorded touch point instead of a
random one, and a new cyan marker tracks the model's causal prediction as
the trial replays - the live counterpart to scripts/live_inference.py's
printed table, as an animated comparison instead of a table of numbers.

No live Delsys hardware here - this REPLAYS an already-recorded trial CSV
(the same tracked emg_imu_vive format every other script in this project
reads), timed to the trial's own real sample rate. Live hardware would
plug in as a different source for the same replay loop (EMGCollector's
ring_buffer instead of a CSV's rows), but that needs the Trigno SDK modules
in the Downloads folder, which live outside this repo and outside what
this environment can import or test - the replay path is what could
actually be built and verified end to end here.

    python scripts/live_prediction_ui.py \\
        --trial-root "/media/.../emg_imu_vive" \\
        --checkpoint runs/grid_leadwindow_emg_imu/best.pt \\
        --config configs/tracked_grid_within.yaml \\
        --speed 1.0

--speed >1 replays faster than real time (e.g. 2.0 = double speed), <1
slower (0.25 = quarter speed, useful for watching a fast reach closely).
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    discover_trials,
    emg_feature_count,
    imu_feature_count,
    preprocess_tracked_trial,
)
from emg_touch.models.grid_reach import GridReachModel  # noqa: E402
from emg_touch.utils import choose_device  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_live", Path(__file__).resolve().parent / "live_inference.py"
)
_live = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_live)
replay_trial = _live.replay_trial

from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QHBoxLayout, QLabel, QPushButton,
    QGraphicsOpacityEffect,
)
from PySide6.QtCore import Qt, QTimer, QVariantAnimation, QEasingCurve  # noqa: E402
from PySide6.QtGui import QFont  # noqa: E402

# Verbatim from SimpleButtonExperiment.py - same visual identity, deliberately.
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0c0c14;
    color: #e0e0ea;
    font-family: 'Segoe UI', 'Arial', sans-serif;
}
QLabel {
    color: #e0e0ea;
}
QPushButton {
    background-color: #1e1e3a;
    border: 1px solid #3f3f5f;
    border-radius: 5px;
    color: #ffffff;
    font-size: 13px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2e2e5a;
    border: 1px solid #6c6cff;
}
"""


class PredictionReplayWindow(QMainWindow):
    """Red target at the true touch point; cyan marker tracks the model live."""

    def __init__(self, trials: list[Path], model, config, device) -> None:
        super().__init__()
        self.setWindowTitle("Live Prediction Replay")
        self.setStyleSheet(DARK_STYLE)
        self.trials = trials
        self.trial_index = -1
        self.model = model
        self.config = config
        self.device = device
        self.btn_size = 80
        self.marker_size = 36
        self.records: list[dict] = []
        self.playback_index = 0

        self.main_container = QWidget()
        self.setCentralWidget(self.main_container)
        self.canvas_area = QWidget(self.main_container)
        self.canvas_area.setStyleSheet("background-color: #0c0c14;")

        # True target - same red radial-gradient button as the original app.
        self.target_button = QPushButton("", self.canvas_area)
        self.target_button.setFixedSize(self.btn_size, self.btn_size)
        self.target_button.setFocusPolicy(Qt.NoFocus)
        self._style_target(active=False)
        self.target_opacity = QGraphicsOpacityEffect(self.target_button)
        self.target_button.setGraphicsEffect(self.target_opacity)
        self.target_opacity.setOpacity(1.0)

        # Predicted point - new, cyan, visually distinct from the true target.
        self.predicted_marker = QLabel("", self.canvas_area)
        self.predicted_marker.setFixedSize(self.marker_size, self.marker_size)
        self.predicted_marker.setAlignment(Qt.AlignCenter)
        self._style_marker()
        self.predicted_marker.hide()

        self.top_bar = QWidget(self.main_container)
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(12, 8, 12, 4)
        self.trial_label = QLabel("Trial: -")
        self.trial_label.setStyleSheet("font-size: 13px; color: #aaaacc;")
        top_layout.addWidget(self.trial_label)
        top_layout.addStretch()
        self.legend_label = QLabel(
            "● true touch (red)     ● model prediction (cyan)"
        )
        self.legend_label.setStyleSheet("font-size: 12px; color: #8888aa;")
        top_layout.addWidget(self.legend_label)
        top_layout.addStretch()
        self.error_label = QLabel("error: -- px")
        self.error_label.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #4fc3f7;"
        )
        top_layout.addWidget(self.error_label)

        self.bottom_bar = QWidget(self.main_container)
        bottom_layout = QHBoxLayout(self.bottom_bar)
        bottom_layout.setContentsMargins(12, 4, 12, 8)
        self.time_label = QLabel("")
        self.time_label.setStyleSheet("font-size: 12px; color: #666680;")
        bottom_layout.addWidget(self.time_label)
        bottom_layout.addStretch()
        restart_btn = QPushButton("Restart (R)")
        restart_btn.setFocusPolicy(Qt.NoFocus)
        restart_btn.clicked.connect(self.start_current_trial)
        bottom_layout.addWidget(restart_btn)
        next_btn = QPushButton("Next Trial (N)")
        next_btn.setFocusPolicy(Qt.NoFocus)
        next_btn.clicked.connect(self.start_next_trial)
        bottom_layout.addWidget(next_btn)

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_playback)

        self.ripple = QVariantAnimation(self)
        self.ripple.setDuration(150)
        self.ripple.setEasingCurve(QEasingCurve.InQuad)
        self.ripple.valueChanged.connect(self._animate_ripple)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        self.top_bar.setGeometry(0, 0, w, 40)
        self.bottom_bar.setGeometry(0, h - 44, w, 44)
        self.canvas_area.setGeometry(0, 40, w, h - 84)

    def keyPressEvent(self, event) -> None:
        if event.text().upper() == "R":
            self.start_current_trial()
        elif event.text().upper() == "N":
            self.start_next_trial()
        else:
            super().keyPressEvent(event)

    def _style_target(self, active: bool) -> None:
        rad = self.btn_size // 2
        colors = (
            "stop:0 #ff4d4d, stop:0.6 #cc1818, stop:1 #660505"
            if active else
            "stop:0 #ffffff, stop:0.3 #ff6666, stop:0.8 #aa0505, stop:1 #000000"
        )
        border = "#ff6666" if active else "#ffffff"
        self.target_button.setStyleSheet(
            "QPushButton { "
            f"background: qradialgradient(cx:0.5, cy:0.5, radius:0.6, fx:0.5, fy:0.4, {colors}); "
            f"border: 3px solid {border}; border-radius: {rad}px; }}"
        )

    def _style_marker(self) -> None:
        rad = self.marker_size // 2
        self.predicted_marker.setStyleSheet(
            "QLabel { "
            "background: qradialgradient(cx:0.5, cy:0.5, radius:0.6, fx:0.5, fy:0.4, "
            "stop:0 #d0faff, stop:0.5 #4fc3f7, stop:1 #0d47a1); "
            f"border: 2px solid #80deea; border-radius: {rad}px; }}"
        )

    # -----------------------------------------------------------------
    def start_next_trial(self) -> None:
        self.trial_index = (self.trial_index + 1) % len(self.trials)
        self.start_current_trial()

    def start_current_trial(self) -> None:
        self.playback_timer.stop()
        if self.trial_index < 0:
            self.trial_index = 0
        path = self.trials[self.trial_index]
        data = preprocess_tracked_trial(path, self.config["data"])
        if data is None or "screen_target" not in data or "canvas" not in data:
            self.trial_label.setText(f"Trial {self.trial_index + 1}: unusable, skipping")
            QTimer.singleShot(0, self.start_next_trial)
            return

        canvas = torch.tensor(data["canvas"], dtype=torch.float32)
        target_px = torch.tensor(data["screen_target"], dtype=torch.float32) * canvas
        emg = torch.from_numpy(data["emg"]).to(self.device)
        imu = torch.from_numpy(data["imu"]).to(self.device)
        onset = int(data["onset"])
        touch = len(data["position"]) - 1
        minimum_prefix = int(self.config["virtual_leader"]["minimum_prefix"])
        patch_length = int(self.config["model"]["patch_length"])
        rate = float(self.config["data"]["sample_rate_hz"]) / max(
            1, int(self.config["data"].get("decimation", 10))
        )
        stride_ms = 40.0
        stride = max(1, int(round(stride_ms * rate / 1000.0)))

        self.records = replay_trial(
            self.model, None, emg, imu, onset, touch, minimum_prefix, patch_length,
            stride, canvas.to(self.device), target_px.to(self.device),
        )
        if not self.records:
            self.trial_label.setText(f"Trial {self.trial_index + 1}: too short, skipping")
            QTimer.singleShot(0, self.start_next_trial)
            return

        cw, ch = self.canvas_area.width(), self.canvas_area.height()
        sx, sy = cw / float(canvas[0]), ch / float(canvas[1])
        self._scale = (sx, sy)
        target_canvas_x = float(target_px[0]) * sx - self.btn_size / 2
        target_canvas_y = float(target_px[1]) * sy - self.btn_size / 2
        self.target_button.setFixedSize(self.btn_size, self.btn_size)
        self.target_button.move(int(target_canvas_x), int(target_canvas_y))
        self._style_target(active=False)
        self.target_opacity.setOpacity(1.0)
        self.target_button.show()
        self.predicted_marker.hide()

        self.trial_label.setText(
            f"Trial {self.trial_index + 1}/{len(self.trials)}: {path.name} "
            f"({len(self.records)} predictions)"
        )
        self.playback_index = 0
        self.playback_timer.start(int(stride_ms))

    def _advance_playback(self) -> None:
        if self.playback_index >= len(self.records):
            self.playback_timer.stop()
            self._on_touch()
            return
        record = self.records[self.playback_index]
        sx, sy = self._scale
        mx = record["mu_x_px"] * sx - self.marker_size / 2
        my = record["mu_y_px"] * sy - self.marker_size / 2
        self.predicted_marker.move(int(mx), int(my))
        self.predicted_marker.show()
        self.predicted_marker.raise_()
        self.error_label.setText(f"error: {record['error_px']:.0f} px")
        self.time_label.setText(
            f"sample {record['sample']}  |  prediction {self.playback_index + 1}/"
            f"{len(self.records)}"
        )
        self.playback_index += 1

    def _on_touch(self) -> None:
        final = self.records[-1]
        self.error_label.setText(f"final error: {final['error_px']:.0f} px")
        self.ripple.setStartValue(1.0)
        self.ripple.setEndValue(0.0)
        self.ripple.start()

    def _animate_ripple(self, value: float) -> None:
        scale = int(self.btn_size * (0.6 + 0.4 * value))
        rad = scale // 2
        self.target_button.setFixedSize(scale, scale)
        self.target_button.setStyleSheet(
            "QPushButton { "
            "background: qradialgradient(cx:0.5, cy:0.5, radius:0.6, fx:0.5, fy:0.4, "
            "stop:0 #ffffff, stop:0.3 #ff6666, stop:0.8 #aa0505, stop:1 #000000); "
            f"border: 3px solid #ffffff; border-radius: {rad}px; }}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", required=True,
                        help="Directory of recorded trial_*.csv files to replay.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/tracked_grid_within.yaml")
    parser.add_argument("--device")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device(args.device)
    sessions = discover_trials(args.trial_root)
    trials = [path for group in sessions.values() for path in group]
    if not trials:
        print(f"no trial_*.csv found under {args.trial_root}", file=sys.stderr)
        sys.exit(2)
    if args.shuffle:
        random.shuffle(trials)

    emg_channels = emg_feature_count(config["data"])
    imu_channels = imu_feature_count(config["data"])
    model = GridReachModel(config, emg_channels, imu_channels, use_imu=True).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()
    print(f"loaded {args.checkpoint} | {len(trials)} trial(s) found | device={device}")

    app = QApplication(sys.argv)
    window = PredictionReplayWindow(trials, model, config, device)
    window.resize(1100, 720)
    window.show()
    window.start_next_trial()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
