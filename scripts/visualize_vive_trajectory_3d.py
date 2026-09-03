#!/usr/bin/env python3
"""Autoplay complete measured VIVE reaches in 3D, without a manipulator.

This is deliberately a measurement viewer, not a model viewer.  It loads the
same held-out split and preprocessing configuration recorded in a training
checkpoint, then animates every measured tracker sample from detected movement
onset to touch.  EMG and IMU are not used to generate the displayed path.

Example:

    python scripts/visualize_vive_trajectory_3d.py \
      --root "/media/.../emg_imu_vive" \
      --checkpoint runs/teacher_bridge/best.pt \
      --cache-dir artifacts/tracked_cache_posture \
      --split test --num-trials 4 --save-gif

On a headless/SSH system, add ``--no-show``.  The animated GIFs and final-frame
PNGs are still written to ``--output-dir``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts import train_latent_distillation_model as training  # noqa: E402
from scripts.visualize_wearable_manipulator import transform_axes  # noqa: E402


def prepare_vive_trajectory(
    position: np.ndarray,
    onset: int,
    sample_rate_hz: float,
    maximum_frames: int,
    relative_to_onset: bool = True,
    include_pre_onset: bool = False,
) -> dict[str, Any]:
    """Clean, optionally downsample, and describe one measured VIVE path."""
    points = np.asarray(position, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"position must have shape [time, 3], got {points.shape}")
    if len(points) < 2:
        raise ValueError("a VIVE trajectory needs at least two samples")
    if not np.isfinite(points).all():
        raise ValueError("VIVE trajectory contains NaN or infinity")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if maximum_frames < 2:
        raise ValueError("maximum_frames must be at least 2")

    onset = int(np.clip(onset, 0, len(points) - 2))
    start = 0 if include_pre_onset else onset
    full_path = points[start:]
    full_indices = np.arange(start, len(points), dtype=np.int64)
    full_elapsed_s = (full_indices - start) / float(sample_rate_hz)
    full_segment_length = np.linalg.norm(np.diff(full_path, axis=0), axis=1)
    full_cumulative_distance = np.concatenate(
        ([0.0], np.cumsum(full_segment_length))
    )
    full_delta_t = np.diff(full_elapsed_s)
    full_speed = np.concatenate((
        [0.0],
        np.divide(
            full_segment_length,
            full_delta_t,
            out=np.zeros_like(full_segment_length),
            where=full_delta_t > 0.0,
        ),
    ))

    selected = np.arange(len(full_path), dtype=np.int64)
    if len(full_path) > maximum_frames:
        selected = np.unique(
            np.rint(np.linspace(0, len(full_path) - 1, maximum_frames)).astype(int)
        )
    display = full_path[selected].copy()
    origin = points[onset].copy()
    if relative_to_onset:
        display -= origin
    onset_point = np.zeros(3, dtype=np.float64) if relative_to_onset else origin
    duration_s = float((len(points) - 1 - start) / sample_rate_hz)
    return {
        "points": display,
        "time_s": full_elapsed_s[selected],
        "speed_mps": full_speed[selected],
        "cumulative_distance_m": full_cumulative_distance[selected],
        "path_length_m": float(full_cumulative_distance[-1]),
        "displacement_m": float(np.linalg.norm(full_path[-1] - full_path[0])),
        "duration_s": duration_s,
        "onset": onset,
        "start": start,
        "origin": origin,
        "onset_point": onset_point,
        "original_sample_count": int(len(points) - start),
    }


def load_checkpoint_config(checkpoint: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"checkpoint {checkpoint} does not contain a config")
    return payload["config"]


def collect_vive_trials(
    loader: torch.utils.data.DataLoader,
    count: int,
    sample_rate_hz: float,
    maximum_frames: int,
    axis_order: str,
    axis_signs: tuple[float, float, float],
    absolute_world: bool,
    include_pre_onset: bool,
) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for batch in loader:
        if batch is None:
            continue
        for row in range(len(batch["lengths"])):
            length = int(batch["lengths"][row])
            onset = int(batch["onset"][row])
            if length < 2 or (not include_pre_onset and onset >= length - 1):
                continue
            measured = transform_axes(
                batch["position"][row, :length].detach().cpu().numpy(),
                axis_order,
                axis_signs,
            )
            prepared = prepare_vive_trajectory(
                measured,
                onset,
                sample_rate_hz,
                maximum_frames,
                relative_to_onset=not absolute_world,
                include_pre_onset=include_pre_onset,
            )
            source = str(batch["paths"][row])
            prepared.update({"path": source, "label": Path(source).stem})
            trials.append(prepared)
            if len(trials) >= count:
                return trials
    return trials


def _set_line_3d(line: Any, points: np.ndarray) -> None:
    line.set_data(points[:, 0], points[:, 1])
    line.set_3d_properties(points[:, 2])


def _set_equal_limits(axis: Any, trials: list[dict[str, Any]]) -> None:
    all_points = np.concatenate([trial["points"] for trial in trials], axis=0)
    low = all_points.min(axis=0)
    high = all_points.max(axis=0)
    centre = 0.5 * (low + high)
    span = max(float(np.ptp(all_points, axis=0).max()), 0.05)
    half = 0.58 * span
    axis.set_xlim(centre[0] - half, centre[0] + half)
    axis.set_ylim(centre[1] - half, centre[1] + half)
    axis.set_zlim(centre[2] - half, centre[2] + half)
    axis.set_box_aspect((1, 1, 1))


def run_viewer(
    trials: list[dict[str, Any]],
    output_dir: Path,
    show: bool,
    save_gif: bool,
    fps: float,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    interval_ms = max(20, int(round(1000.0 / fps)))
    figure = plt.figure(figsize=(13.5, 7.8))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.55, 1.0))
    path_axis = figure.add_subplot(grid[0, 0], projection="3d")
    motion_axis = figure.add_subplot(grid[0, 1])
    figure.subplots_adjust(bottom=0.18, wspace=0.27)

    path_axis.set_xlabel("VIVE x (m)")
    path_axis.set_ylabel("VIVE y (m)")
    path_axis.set_zlabel("VIVE z (m)")
    _set_equal_limits(path_axis, trials)
    (trail,) = path_axis.plot(
        [], [], [], color="#167C80", linewidth=3.0, label="measured VIVE trail"
    )
    (current,) = path_axis.plot(
        [], [], [], marker="o", markersize=9, color="#E4572E", linestyle="None",
        label="current tracker position",
    )
    (onset_marker,) = path_axis.plot(
        [], [], [], marker="s", markersize=8, color="#315A7D", linestyle="None",
        label="movement start",
    )
    (touch_marker,) = path_axis.plot(
        [], [], [], marker="*", markersize=14, color="#D4A017", linestyle="None",
        label="touch",
    )
    path_axis.legend(loc="upper left", fontsize=8)

    (speed_line,) = motion_axis.plot([], [], color="#8C4A9E", label="tracker speed")
    motion_cursor = motion_axis.axvline(0.0, color="0.25", linewidth=1)
    motion_axis.set_xlabel("time from displayed start (s)")
    motion_axis.set_ylabel("speed (m/s)")
    motion_axis.set_title("measured motion as the 3D trail grows")
    motion_axis.grid(alpha=0.2)
    motion_axis.legend(fontsize=8)

    state = {"trial": 0, "frame": 0, "playing": bool(show)}
    trial_slider_axis = figure.add_axes((0.12, 0.09, 0.30, 0.028))
    frame_slider_axis = figure.add_axes((0.12, 0.045, 0.55, 0.028))
    play_axis = figure.add_axes((0.73, 0.075, 0.08, 0.05))
    next_axis = figure.add_axes((0.83, 0.075, 0.08, 0.05))
    trial_slider = Slider(
        trial_slider_axis, "trial", 0, max(1, len(trials) - 1),
        valinit=0, valstep=1,
    )
    frame_slider = Slider(
        frame_slider_axis, "sample", 0, max(1, len(trials[0]["points"]) - 1),
        valinit=0, valstep=1,
    )
    play_button = Button(play_axis, "Pause" if show else "Play")
    next_button = Button(next_axis, "Next trial")
    summary = figure.text(0.51, 0.965, "", ha="center", va="center", fontsize=9)

    def redraw() -> None:
        trial = trials[state["trial"]]
        frame = min(state["frame"], len(trial["points"]) - 1)
        points = trial["points"]
        time_s = trial["time_s"]
        _set_line_3d(trail, points[: frame + 1])
        _set_line_3d(current, points[frame : frame + 1])
        _set_line_3d(onset_marker, trial["onset_point"][None, :])
        _set_line_3d(touch_marker, points[-1:])
        speed_line.set_data(time_s[: frame + 1], trial["speed_mps"][: frame + 1])
        motion_axis.set_xlim(0.0, max(float(time_s[-1]), 0.01))
        speed_max = max(float(np.max(trial["speed_mps"])), 0.05)
        motion_axis.set_ylim(0.0, 1.08 * speed_max)
        motion_cursor.set_xdata([time_s[frame], time_s[frame]])
        path_axis.set_title(
            f"{trial['label']} — measured sample {frame + 1}/{len(points)}"
        )
        summary.set_text(
            f"t={time_s[frame]:.2f}/{trial['duration_s']:.2f} s  |  "
            f"distance travelled={trial['cumulative_distance_m'][frame] * 100:.1f} cm  |  "
            f"full path={trial['path_length_m'] * 100:.1f} cm  |  "
            f"start→touch={trial['displacement_m'] * 100:.1f} cm"
        )
        figure.canvas.draw_idle()

    def select_trial(value: float) -> None:
        state["trial"] = min(int(value), len(trials) - 1)
        state["frame"] = 0
        maximum = len(trials[state["trial"]]["points"]) - 1
        frame_slider.valmax = maximum
        frame_slider.ax.set_xlim(0, max(1, maximum))
        frame_slider.set_val(0)
        redraw()

    def select_frame(value: float) -> None:
        state["frame"] = min(int(value), len(trials[state["trial"]]["points"]) - 1)
        redraw()

    def toggle_play(_: Any) -> None:
        state["playing"] = not state["playing"]
        play_button.label.set_text("Pause" if state["playing"] else "Play")

    def next_trial(_: Any) -> None:
        trial_slider.set_val((state["trial"] + 1) % len(trials))
        state["playing"] = True
        play_button.label.set_text("Pause")

    def timer_step() -> None:
        if not state["playing"]:
            return
        maximum = len(trials[state["trial"]]["points"]) - 1
        if state["frame"] >= maximum:
            state["playing"] = False
            play_button.label.set_text("Play")
            return
        frame_slider.set_val(state["frame"] + 1)

    trial_slider.on_changed(select_trial)
    frame_slider.on_changed(select_frame)
    play_button.on_clicked(toggle_play)
    next_button.on_clicked(next_trial)
    timer = figure.canvas.new_timer(interval=interval_ms)
    timer.add_callback(timer_step)
    timer.start()

    output_dir.mkdir(parents=True, exist_ok=True)
    redraw()
    original_trial, original_frame = state["trial"], state["frame"]
    for index, trial in enumerate(trials):
        state["trial"] = index
        state["frame"] = len(trial["points"]) - 1
        redraw()
        stem = f"vive_{index:02d}_{trial['label']}"
        figure.savefig(output_dir / f"{stem}.png", dpi=130)
        if save_gif:
            from matplotlib.animation import FuncAnimation, PillowWriter

            def gif_frame(frame: int, trial_index: int = index) -> None:
                state["trial"] = trial_index
                state["frame"] = frame
                redraw()

            animation = FuncAnimation(
                figure,
                gif_frame,
                frames=len(trial["points"]),
                interval=interval_ms,
                repeat=True,
                blit=False,
            )
            animation.save(
                output_dir / f"{stem}.gif",
                writer=PillowWriter(fps=fps),
                dpi=90,
            )
    state["trial"], state["frame"] = original_trial, original_frame
    redraw()
    print(f"wrote {len(trials)} VIVE snapshot(s) to {output_dir.resolve()}")
    if save_gif:
        print(f"wrote {len(trials)} full-trajectory animated GIF(s)")
    if show:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache_posture")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--axis-order", default="xyz",
        help="Displayed x/y/z source axes as a VIVE-axis permutation",
    )
    parser.add_argument(
        "--axis-signs", type=float, nargs=3, default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
    )
    parser.add_argument(
        "--absolute-world", action="store_true",
        help="Show VIVE world coordinates instead of making onset [0,0,0]",
    )
    parser.add_argument(
        "--include-pre-onset", action="store_true",
        help="Animate the entire recorded trial, including rest before onset",
    )
    parser.add_argument("--output-dir", default="runs/vive_trajectory_3d")
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.num_trials < 1:
        raise SystemExit("--num-trials must be positive")
    if args.max_frames < 2:
        raise SystemExit("--max-frames must be at least 2")
    if args.fps <= 0.0:
        raise SystemExit("--fps must be positive")

    config = load_checkpoint_config(args.checkpoint)
    loaders = training.build_experiment_loaders(config, args.root, Path(args.cache_dir))
    loader = loaders[{"train": 0, "validation": 1, "test": 2}[args.split]]
    rate = training.effective_rate(config)
    trials = collect_vive_trials(
        loader,
        args.num_trials,
        rate,
        args.max_frames,
        args.axis_order.lower(),
        tuple(args.axis_signs),
        args.absolute_world,
        args.include_pre_onset,
    )
    if not trials:
        raise SystemExit("no usable VIVE trajectories found in the selected split")

    scope = "complete recording" if args.include_pre_onset else "movement onset → touch"
    coordinates = "absolute VIVE world" if args.absolute_world else "onset-relative"
    print(
        f"loaded {len(trials)} {args.split} VIVE trial(s) | {scope} | "
        f"{coordinates} | effective rate={rate:.2f} Hz"
    )
    print("measurement check: this viewer draws VIVE only; no model or manipulator")
    for index, trial in enumerate(trials):
        print(
            f"  {index}: {trial['label']} | duration={trial['duration_s']:.2f}s | "
            f"path={trial['path_length_m'] * 100:.1f}cm | "
            f"start→touch={trial['displacement_m'] * 100:.1f}cm | "
            f"display frames={len(trial['points'])}/"
            f"{trial['original_sample_count']}"
        )
    run_viewer(
        trials,
        Path(args.output_dir),
        show=not args.no_show,
        save_gif=args.save_gif,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
