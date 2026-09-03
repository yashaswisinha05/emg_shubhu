#!/usr/bin/env python3
"""Drive a 3R manipulator with an actual wearable-model 3D prediction.

The base and shoulder point P1 are the same origin.  By default, the causal
EMG+IMU student is queried at movement onset for its complete onset-to-touch
trajectory.  The orange manipulator follows that prediction through analytical
inverse kinematics; VIVE is drawn only as a black held-out comparison path.

The model was trained on 50--400 ms horizons.  A full reach longer than that is
a genuine wearable-model output, but it is an out-of-training-range
extrapolation and is labelled as such in the viewer.  Use
``--trajectory-window fixed-horizon --lead-ms 200`` to remain inside the
trained horizon.

Without ``--base-world``, the relative reach is anchored to a configurable
synthetic initial arm pose.  This is sufficient to inspect motion shape.  For
physical joint angles, pass the measured VIVE-world XYZ of the shoulder/base.

Example:

    python scripts/visualize_wearable_manipulator.py \
      --root "/media/.../emg_imu_vive" \
      --checkpoint runs/teacher_bridge/best.pt \
      --cache-dir artifacts/tracked_cache_posture \
      --device cuda --lead-ms 200 --num-trials 4

Measured shoulder calibration example:

    ... --base-world 0.42 -0.18 1.07 --axis-order xyz --axis-signs 1 1 1
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
from emg_touch.live_distillation import LiveDistillationModel  # noqa: E402
from emg_touch.physics.manipulator_ik import ThreeRManipulator  # noqa: E402


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def transform_axes(
    values: np.ndarray, order: str, signs: tuple[float, float, float]
) -> np.ndarray:
    """Map VIVE XYZ vectors into the manipulator coordinate convention."""
    if len(order) != 3 or set(order) != {"x", "y", "z"}:
        raise ValueError("axis order must be a permutation such as xyz or xzy")
    indices = [AXIS_INDEX[label] for label in order]
    return np.asarray(values)[..., indices] * np.asarray(signs)


def one_row(batch: dict[str, Any], row: int) -> dict[str, Any]:
    """Slice one collated trial while preserving non-batched metadata."""
    batch_size = int(len(batch["lengths"]))
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == batch_size:
            result[key] = value[row : row + 1]
        elif isinstance(value, list) and len(value) == batch_size:
            result[key] = [value[row]]
        else:
            result[key] = value
    return result


@torch.inference_mode()
def collect_trials(
    runner: LiveDistillationModel,
    loader: torch.utils.data.DataLoader,
    manipulator: ThreeRManipulator,
    count: int,
    lead_samples: int,
    initial_angles: np.ndarray,
    base_world: np.ndarray | None,
    axis_order: str,
    axis_signs: tuple[float, float, float],
    trajectory_window: str,
) -> list[dict[str, Any]]:
    config = runner.config
    model = runner.model
    device = runner.device
    teacher_steps = int(config["model"]["teacher_trajectory_steps"])
    trajectory_limit = float(config["model"].get("trajectory_limit_m", 0.8))
    velocity_scale = float(config["model"].get("velocity_scale_mps", 1.0))
    rate = training.effective_rate(config)
    trained_window_ms = config.get("distillation", {}).get(
        "lead_window_ms", [50.0, 400.0]
    )
    trained_max_ms = float(max(trained_window_ms))
    generator = np.random.default_rng(0)
    trials: list[dict[str, Any]] = []

    for raw_batch in loader:
        if raw_batch is None:
            continue
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        for row in range(len(batch["lengths"])):
            length = int(batch["lengths"][row])
            touch = length - 1
            if touch <= 1:
                continue
            onset = int(np.clip(int(batch["onset"][row]), 1, touch - 1))
            selected_lead = (
                touch - onset
                if trajectory_window == "full-reach"
                else int(lead_samples)
            )
            cut = touch - selected_lead
            if cut <= 0:
                continue
            single = one_row(batch, row)
            window = training.make_distillation_window(
                single,
                runner.context_samples,
                runner.patch_length,
                teacher_steps,
                generator,
                trajectory_limit,
                velocity_scale,
                fallback_canvas=None,
                fixed_lead=selected_lead,
            )
            if window is None:
                continue
            output = model.student_forward(
                window["emg"],
                window["imu"],
                window["time_mask"],
                sample=False,
            )
            predicted_relative = transform_axes(
                output["trajectory"][0].detach().cpu().numpy(),
                axis_order,
                axis_signs,
            )
            true_relative = transform_axes(
                window["trajectory_target"][0].detach().cpu().numpy(),
                axis_order,
                axis_signs,
            )
            vive_world = batch["position"][row].detach().cpu().numpy()
            if base_world is None:
                start_hand = manipulator.forward(initial_angles)[-1]
                calibration = "synthetic initial pose"
                starting_angles = initial_angles
                initial_was_projected = False
            else:
                tracker_origin = vive_world[cut - 1]
                start_hand = transform_axes(
                    tracker_origin - base_world, axis_order, axis_signs
                )
                initial_solution = manipulator.inverse(
                    start_hand, previous=initial_angles
                )
                starting_angles = initial_solution.angles
                start_hand = initial_solution.projected
                calibration = "measured shoulder/base"
                initial_was_projected = initial_solution.was_projected

            # Prepend the common origin so the animation begins at the actual
            # cutoff pose.  Every subsequent orange-arm target comes from the
            # wearable student's trajectory head.
            predicted_path = np.concatenate(
                [start_hand[None, :], start_hand[None, :] + predicted_relative],
                axis=0,
            )
            vive_path = np.concatenate(
                [start_hand[None, :], start_hand[None, :] + true_relative],
                axis=0,
            )
            followed = manipulator.follow(
                predicted_path, initial_angles=starting_angles
            )
            point_error_cm = 100.0 * np.linalg.norm(
                predicted_path - vive_path, axis=-1
            )
            ik_error_cm = 100.0 * np.linalg.norm(
                followed["chain"][:, -1] - predicted_path, axis=-1
            )
            horizon_ms = 1000.0 * selected_lead / rate
            time_ms = np.concatenate((
                [0.0],
                np.linspace(
                    1000.0 / rate,
                    1000.0 * (selected_lead + 1) / rate,
                    teacher_steps,
                ),
            ))
            source = window["source_paths"][0]
            trials.append({
                "path": source,
                "label": Path(source).stem,
                "calibration": calibration,
                "initial_was_projected": initial_was_projected,
                "vive_path": vive_path,
                "predicted_path": predicted_path,
                "forecast_true_path": vive_path,
                "followed": followed,
                "model_followed": followed,
                "model_error_cm": point_error_cm,
                "ik_error_cm": ik_error_cm,
                "time_ms": time_ms,
                "forecast_time_ms": time_ms,
                "forecast_start_ms": 0.0,
                "lead_ms": horizon_ms,
                "trained_max_ms": trained_max_ms,
                "out_of_range": horizon_ms > trained_max_ms + 1e-6,
                "trajectory_window": trajectory_window,
                "cut_samples_past_onset": cut - onset,
            })
            if len(trials) >= count:
                return trials
    return trials


def _set_line_3d(line: Any, points: np.ndarray) -> None:
    line.set_data(points[:, 0], points[:, 1])
    line.set_3d_properties(points[:, 2])


def _equal_limits(axis: Any, trials: list[dict[str, Any]], reach: float) -> None:
    points = [np.zeros((1, 3))]
    for trial in trials:
        points.extend([
            trial["vive_path"],
            trial["predicted_path"],
            trial["forecast_true_path"],
            trial["followed"]["chain"].reshape(-1, 3),
        ])
        if "explicit_endpoint" in trial:
            points.append(np.asarray(trial["explicit_endpoint"])[None, :])
    all_points = np.concatenate(points)
    centre = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    span = max(float(np.ptp(all_points, axis=0).max()), 0.5 * reach)
    half = 0.58 * span
    axis.set_xlim(centre[0] - half, centre[0] + half)
    axis.set_ylim(centre[1] - half, centre[1] + half)
    axis.set_zlim(centre[2] - half, centre[2] + half)
    axis.set_box_aspect((1, 1, 1))


def run_viewer(
    trials: list[dict[str, Any]],
    manipulator: ThreeRManipulator,
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

    figure = plt.figure(figsize=(14, 8))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.55, 1.0), height_ratios=(1.0, 1.0)
    )
    arm_axis = figure.add_subplot(grid[:, 0], projection="3d")
    joint_axis = figure.add_subplot(grid[0, 1])
    error_axis = figure.add_subplot(grid[1, 1])
    figure.subplots_adjust(bottom=0.18, wspace=0.25, hspace=0.32)

    arm_axis.set_xlabel("manipulator x / reach (m)")
    arm_axis.set_ylabel("manipulator y / lateral (m)")
    arm_axis.set_zlabel("manipulator z / vertical (m)")
    _equal_limits(arm_axis, trials, manipulator.maximum_reach)
    arm_axis.scatter(
        [0.0], [0.0], [0.0], s=90, marker="s", color="#7A1F1F", label="base = P1"
    )
    arm_axis.text(0.0, 0.0, 0.0, "  base = P1")
    (true_line,) = arm_axis.plot(
        [], [], [], color="#222222", linewidth=2.2,
        label="VIVE ground truth (comparison only)"
    )
    (prediction_line,) = arm_axis.plot(
        [], [], [], "--", color="#17A6B6", linewidth=2.2,
        label=trials[0].get("prediction_label", "EMG+IMU forecast"),
    )
    (executed_line,) = arm_axis.plot(
        [], [], [], ":", color="#2E8B57", linewidth=2.2,
        label="IK/FK following model",
    )
    (arm_line,) = arm_axis.plot(
        [], [], [], "-o", color="#D47A20", linewidth=5, markersize=7,
        label="3R manipulator (EMG+IMU-driven)",
    )
    (true_current,) = arm_axis.plot(
        [], [], [], marker="x", color="#222222", markersize=10, linestyle="None"
    )
    (predicted_current,) = arm_axis.plot(
        [], [], [], marker="o", color="#17A6B6", markersize=7, linestyle="None"
    )
    (endpoint_marker,) = arm_axis.plot(
        [], [], [], marker="*", color="#8B3FC7", markersize=12,
        linestyle="None", label="explicit model 3D endpoint",
    )
    arm_axis.legend(loc="upper left", fontsize=8)

    joint_lines = [
        joint_axis.plot([], [], label=label)[0]
        for label in ("q1 yaw", "q2 shoulder", "q3 elbow")
    ]
    joint_cursor = joint_axis.axvline(0.0, color="0.25", linewidth=1)
    joint_axis.set_ylabel("joint angle (degrees)")
    joint_axis.set_xlabel(
        trials[0].get("time_axis_label", "time from model cutoff (ms)")
    )
    joint_axis.set_title("model-driven inverse-kinematics joint trajectory")
    joint_axis.legend(fontsize=8)

    (model_error_line,) = error_axis.plot(
        [], [], color="#17A6B6",
        label=trials[0].get("model_error_label", "model forecast vs VIVE"),
    )
    (ik_error_line,) = error_axis.plot(
        [], [], color="#2E8B57", label="IK/FK vs model request"
    )
    error_cursor = error_axis.axvline(0.0, color="0.25", linewidth=1)
    error_axis.set_xlabel(
        trials[0].get("time_axis_label", "time from model cutoff (ms)")
    )
    error_axis.set_ylabel("3D Euclidean error (cm)")
    error_axis.set_title("trajectory and reachability errors")
    error_axis.legend(fontsize=8)

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
        frame_slider_axis, "IK step", 0,
        max(0, len(trials[0]["time_ms"]) - 1), valinit=0, valstep=1,
    )
    play_button = Button(play_axis, "Pause" if show else "Play")
    next_button = Button(next_axis, "Next trial")

    summary = figure.text(0.51, 0.965, "", ha="center", va="center", fontsize=9)

    def redraw() -> None:
        trial = trials[state["trial"]]
        frame = min(state["frame"], len(trial["time_ms"]) - 1)
        followed = trial["followed"]
        # VIVE and model paths share a timeline, but only the cyan model path
        # supplies targets to the orange manipulator.
        _set_line_3d(true_line, trial["vive_path"][: frame + 1])
        _set_line_3d(prediction_line, trial["predicted_path"][: frame + 1])
        _set_line_3d(executed_line, followed["chain"][: frame + 1, -1])
        _set_line_3d(arm_line, followed["chain"][frame])
        _set_line_3d(true_current, trial["vive_path"][frame : frame + 1])
        _set_line_3d(
            predicted_current, trial["predicted_path"][frame : frame + 1]
        )
        if "explicit_endpoint" in trial:
            endpoint = np.asarray(trial["explicit_endpoint"])[None, :]
            _set_line_3d(endpoint_marker, endpoint)
        else:
            _set_line_3d(endpoint_marker, np.empty((0, 3)))

        time = trial["time_ms"]
        angles_deg = np.rad2deg(followed["angles"])
        for index, line in enumerate(joint_lines):
            line.set_data(time[: frame + 1], angles_deg[: frame + 1, index])
        joint_axis.set_xlim(0.0, max(float(time[-1]), 1.0))
        angle_low = float(angles_deg.min())
        angle_high = float(angles_deg.max())
        angle_margin = max(3.0, 0.08 * (angle_high - angle_low))
        joint_axis.set_ylim(angle_low - angle_margin, angle_high + angle_margin)
        joint_cursor.set_xdata([time[frame], time[frame]])

        model_error_line.set_data(
            time[: frame + 1], trial["model_error_cm"][: frame + 1],
        )
        ik_error_line.set_data(
            time[: frame + 1], trial["ik_error_cm"][: frame + 1]
        )
        error_axis.set_xlim(0.0, max(float(time[-1]), 1.0))
        error_high = max(
            float(np.max(trial["model_error_cm"])),
            float(np.max(trial["ik_error_cm"])),
            0.1,
        )
        error_axis.set_ylim(0.0, 1.08 * error_high)
        error_cursor.set_xdata([time[frame], time[frame]])
        projected = int(followed["was_projected"].sum())
        range_label = (
            f"EXTRAPOLATED beyond {trial['trained_max_ms']:.0f} ms training max"
            if trial["out_of_range"] else "inside trained lead window"
        )
        arm_axis.set_title(
            f"{trial['label']} — model-driven trajectory {frame + 1}/{len(time)} — "
            f"{trial['calibration']}"
        )
        endpoint_text = (
            f"  |  endpoint/path={trial['endpoint_agreement_cm']:.2f} cm"
            if "endpoint_agreement_cm" in trial else ""
        )
        summary.set_text(
            f"EMG+IMU MODEL-DRIVEN IK  |  {range_label}  |  mean model↔VIVE "
            f"{trial['model_error_cm'][1:].mean():.2f} cm  |  "
            f"current={trial['model_error_cm'][frame]:.2f} cm  |  "
            f"model workspace projections={projected}/{len(time)}  |  "
            f"q=[{angles_deg[frame, 0]:.1f}°, {angles_deg[frame, 1]:.1f}°, "
            f"{angles_deg[frame, 2]:.1f}°]{endpoint_text}"
        )
        figure.canvas.draw_idle()

    def select_trial(value: float) -> None:
        state["trial"] = min(int(value), len(trials) - 1)
        state["frame"] = 0
        maximum = max(0, len(trials[state["trial"]]["time_ms"]) - 1)
        frame_slider.valmax = maximum
        frame_slider.ax.set_xlim(0, max(1, maximum))
        frame_slider.set_val(0)
        redraw()

    def select_frame(value: float) -> None:
        state["frame"] = int(value)
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
        maximum = len(trials[state["trial"]]["time_ms"]) - 1
        if state["frame"] >= maximum:
            state["playing"] = False
            play_button.label.set_text("Play")
            return
        frame_slider.set_val(state["frame"] + 1)

    trial_slider.on_changed(select_trial)
    frame_slider.on_changed(select_frame)
    play_button.on_clicked(toggle_play)
    next_button.on_clicked(next_trial)
    interval_ms = max(20, int(round(1000.0 / fps)))
    timer = figure.canvas.new_timer(interval=interval_ms)
    timer.add_callback(timer_step)
    timer.start()

    output_dir.mkdir(parents=True, exist_ok=True)
    redraw()
    original_trial, original_frame = state["trial"], state["frame"]
    for index, trial in enumerate(trials):
        state["trial"] = index
        state["frame"] = len(trial["time_ms"]) - 1
        redraw()
        figure.savefig(
            output_dir / f"manipulator_{index:02d}_{trial['label']}.png",
            dpi=130,
        )
        if save_gif:
            from matplotlib.animation import FuncAnimation, PillowWriter

            def gif_frame(frame: int, trial_index: int = index) -> None:
                state["trial"] = trial_index
                state["frame"] = frame
                redraw()

            animation = FuncAnimation(
                figure,
                gif_frame,
                frames=len(trial["time_ms"]),
                interval=interval_ms,
                repeat=True,
                blit=False,
            )
            animation.save(
                output_dir / f"manipulator_{index:02d}_{trial['label']}.gif",
                writer=PillowWriter(fps=fps),
                dpi=100,
            )
    state["trial"], state["frame"] = original_trial, original_frame
    redraw()
    print(f"wrote {len(trials)} manipulator snapshot(s) to {output_dir.resolve()}")
    if save_gif:
        print(f"wrote {len(trials)} model+IK animated GIF(s)")
    if show:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache_posture")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--lead-ms", type=float, default=200.0)
    parser.add_argument(
        "--trajectory-window",
        choices=("full-reach", "fixed-horizon"),
        default="full-reach",
        help=(
            "full-reach queries the model at movement onset; fixed-horizon "
            "uses --lead-ms before touch"
        ),
    )
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument(
        "--max-frames", type=int, default=180,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--link-lengths", type=float, nargs=2, default=(0.50, 0.60))
    parser.add_argument(
        "--initial-joint-deg", type=float, nargs=3, default=(0.0, 20.0, 90.0),
        metavar=("YAW", "SHOULDER", "ELBOW"),
    )
    parser.add_argument(
        "--base-world", type=float, nargs=3, metavar=("X", "Y", "Z"),
        help="Measured shoulder/base location in VIVE world metres",
    )
    parser.add_argument(
        "--axis-order", default="xyz",
        help="Manipulator x/y/z source axes as a VIVE-axis permutation",
    )
    parser.add_argument(
        "--axis-signs", type=float, nargs=3, default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
    )
    parser.add_argument("--output-dir", default="runs/wearable_manipulator")
    parser.add_argument(
        "--save-gif", action="store_true",
        help="Also export an animated GIF for every selected trial",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.num_trials < 1:
        raise SystemExit("--num-trials must be positive")
    if args.max_frames < 2:
        raise SystemExit("--max-frames must be at least 2")
    if args.fps <= 0.0:
        raise SystemExit("--fps must be positive")

    runner = LiveDistillationModel(
        "wearable manipulator", args.checkpoint, args.device
    )
    loaders = training.build_experiment_loaders(
        runner.config, args.root, Path(args.cache_dir)
    )
    loader = loaders[{"train": 0, "validation": 1, "test": 2}[args.split]]
    rate = training.effective_rate(runner.config)
    lead_samples = max(1, training.milliseconds_to_samples(args.lead_ms, rate))
    manipulator = ThreeRManipulator(tuple(args.link_lengths))
    initial_angles = np.deg2rad(np.asarray(args.initial_joint_deg, dtype=np.float64))
    base_world = (
        None if args.base_world is None
        else np.asarray(args.base_world, dtype=np.float64)
    )
    trials = collect_trials(
        runner,
        loader,
        manipulator,
        args.num_trials,
        lead_samples,
        initial_angles,
        base_world,
        args.axis_order.lower(),
        tuple(args.axis_signs),
        args.trajectory_window,
    )
    if not trials:
        raise SystemExit("no usable trials were long enough for the selected lead")

    print(
        f"loaded {runner.kind} | {len(trials)} {args.split} trial(s) | "
        f"trajectory-window={args.trajectory_window}"
    )
    print("deployment check: model trajectory receives causal EMG+IMU only")
    print(
        "visualisation check: orange arm follows the cyan EMG+IMU model path; "
        "black VIVE is comparison only"
    )
    print("base=P1=[0,0,0] in manipulator coordinates")
    for index, trial in enumerate(trials):
        followed = trial["followed"]
        print(
            f"  {index}: {trial['label']} | horizon={trial['lead_ms']:.1f} ms"
            f"{' OUT-OF-RANGE' if trial['out_of_range'] else ''} | model↔VIVE="
            f"{trial['model_error_cm'][1:].mean():.2f} cm | IK projection="
            f"{int(followed['was_projected'].sum())}/{len(followed['was_projected'])} "
            f"| initial projection={trial['initial_was_projected']} "
            f"| IK/FK residual={trial['ik_error_cm'].mean():.4f} cm"
        )
    run_viewer(
        trials,
        manipulator,
        Path(args.output_dir),
        show=not args.no_show,
        save_gif=args.save_gif,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
