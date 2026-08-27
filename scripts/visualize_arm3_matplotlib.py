#!/usr/bin/env python3
"""Interactive matplotlib 3D viewer for the 3-DOF arm (arm3.ThreeDofArm).

Two modes:

Manual (default, no --checkpoint needed): three sliders for q1 (shoulder,
z-axis), q2 (shoulder, -y axis), q3 (elbow, -y axis), redrawing the
shoulder-elbow-hand chain live - a kinematics sanity check independent of any
trained weights, same purpose as the earlier 2-link HTML manipulator.

Trial playback (--checkpoint given): loads a grid_fusion_physics3 checkpoint,
runs it on a batch of real validation trials, and lets you scrub through the
torque-head rollout's actual joint trajectory with a frame slider, alongside
the click target and both the physics-only and blended screen predictions.

Usage:
  python scripts/visualize_arm3_matplotlib.py
  python scripts/visualize_arm3_matplotlib.py \
    --config configs/hill_fusion.yaml \
    --checkpoint runs/hill_fusion3_cuda/grid_fusion_physics3/best.pt \
    --trials 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, Slider

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emg_touch.physics.arm3 import ThreeDofArm


def _chain_points(arm: ThreeDofArm, angles: torch.Tensor) -> np.ndarray:
    """Shoulder / elbow / hand positions (3, 3) for one angle triple."""
    angles = angles.unsqueeze(0)
    _, t12, _ = arm._fk(angles)
    elbow_home = torch.tensor([float(arm.link_length[0]), 0.0, 0.0, 1.0])
    shoulder = np.zeros(3)
    elbow = (t12[0] @ elbow_home)[:3].numpy()
    hand = arm.endpoint(angles)[0].numpy()
    return np.stack([shoulder, elbow, hand])


def _set_equal_3d(ax, reach: float) -> None:
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_zlim(-reach, reach)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z (gravity axis)")


def run_manual(arm: ThreeDofArm) -> None:
    reach = float(arm.link_length.sum()) * 1.15

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.28)
    _set_equal_3d(ax, reach)
    ax.set_title("3-DOF arm: manual kinematics check")

    (line,) = ax.plot([], [], [], "-o", color="#378ADD", linewidth=4, markersize=8)
    (hand_marker,) = ax.plot([], [], [], "o", color="#D85A30", markersize=12)

    def redraw(q1: float, q2: float, q3: float) -> None:
        angles = torch.tensor([q1, q2, q3])
        points = _chain_points(arm, angles)
        line.set_data(points[:, 0], points[:, 1])
        line.set_3d_properties(points[:, 2])
        hand_marker.set_data([points[2, 0]], [points[2, 1]])
        hand_marker.set_3d_properties([points[2, 2]])
        fig.canvas.draw_idle()

    ax_q1 = plt.axes((0.2, 0.16, 0.6, 0.03))
    ax_q2 = plt.axes((0.2, 0.11, 0.6, 0.03))
    ax_q3 = plt.axes((0.2, 0.06, 0.6, 0.03))
    s_q1 = Slider(ax_q1, "q1 (shoulder, z)", -1.8, 1.8, valinit=0.2)
    s_q2 = Slider(ax_q2, "q2 (shoulder, -y)", -1.8, 1.8, valinit=0.3)
    s_q3 = Slider(ax_q3, "q3 (elbow, -y)", 0.0, 2.8, valinit=1.4)

    def on_change(_) -> None:
        redraw(s_q1.val, s_q2.val, s_q3.val)

    s_q1.on_changed(on_change)
    s_q2.on_changed(on_change)
    s_q3.on_changed(on_change)
    redraw(s_q1.val, s_q2.val, s_q3.val)
    plt.show()


def run_playback(arm: ThreeDofArm, config_path: str, checkpoint_path: str, trials: int) -> None:
    from emg_touch.config import load_config
    from emg_touch.data.grid_trajectory import build_grid_trajectory_loaders
    from emg_touch.models.grid_point import build_grid_model

    config = load_config(config_path)
    model = build_grid_model("grid_fusion_physics3", config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, val_loader, _ = build_grid_trajectory_loaders(
        config,
        split_path=config["paths"]["split_file"],
        scaler_path=config["paths"]["scaler"],
    )
    batch = next(iter(val_loader))
    with torch.no_grad():
        outputs = model(batch)

    decimation = int(config.get("physics", {}).get("decimation", 4))
    trajectory = outputs["physics_trajectory"]
    lengths = batch["lengths"]
    count = min(trials, trajectory.size(0))
    trial_ids = [str(t).split("__")[-1] for t in batch["trial_id"][:count]]
    trial_steps = [
        int((lengths[i].item() + decimation - 1) // decimation) for i in range(count)
    ]

    reach = float(arm.link_length.sum()) * 1.15
    fig = plt.figure(figsize=(11, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)
    plt.subplots_adjust(bottom=0.28)
    _set_equal_3d(ax3d, reach)
    ax2d.set_xlim(0, 1)
    ax2d.set_ylim(1, 0)
    ax2d.set_title("screen prediction vs click target")
    ax2d.set_aspect("equal")

    (line,) = ax3d.plot([], [], [], "-o", color="#378ADD", linewidth=4, markersize=8)
    (hand_marker,) = ax3d.plot([], [], [], "o", color="#D85A30", markersize=10)
    (target_pt,) = ax2d.plot([], [], "o", color="#D85A30", markersize=12, label="click target")
    (fusion_pt,) = ax2d.plot([], [], "o", color="#378ADD", markersize=10, label="fusion prediction")
    (physics_pt,) = ax2d.plot([], [], "o", color="#0F6E56", markersize=10, label="physics prediction")
    ax2d.legend(loc="upper right", fontsize=8)

    state = {"trial": 0, "frame": 0, "playing": False}

    ax_trial = plt.axes((0.15, 0.16, 0.35, 0.03))
    ax_frame = plt.axes((0.15, 0.11, 0.55, 0.03))
    ax_play = plt.axes((0.75, 0.16, 0.1, 0.05))
    s_trial = Slider(ax_trial, "trial", 0, count - 1, valinit=0, valstep=1)
    s_frame = Slider(ax_frame, "step", 0, max(trial_steps[0] - 1, 1), valinit=0, valstep=1)
    b_play = Button(ax_play, "Play")

    def redraw() -> None:
        trial = state["trial"]
        frame = min(state["frame"], trial_steps[trial] - 1)
        angles = trajectory[trial, frame]
        points = _chain_points(arm, angles)
        line.set_data(points[:, 0], points[:, 1])
        line.set_3d_properties(points[:, 2])
        hand_marker.set_data([points[2, 0]], [points[2, 1]])
        hand_marker.set_3d_properties([points[2, 2]])

        target = batch["target"][trial].numpy()
        fusion = outputs["fusion_prediction"][trial].detach().numpy()
        physics = outputs["physics_prediction"][trial].detach().numpy()
        target_pt.set_data([target[0]], [target[1]])
        fusion_pt.set_data([fusion[0]], [fusion[1]])
        physics_pt.set_data([physics[0]], [physics[1]])
        blend = float(outputs["physics_blend"][trial])
        ax3d.set_title(
            f"trial {trial_ids[trial]}  step {frame}/{trial_steps[trial]-1}  "
            f"blend={blend:.4f}"
        )
        fig.canvas.draw_idle()

    def on_trial_change(_) -> None:
        state["trial"] = int(s_trial.val)
        state["frame"] = 0
        s_frame.valmax = max(trial_steps[state["trial"]] - 1, 1)
        s_frame.ax.set_xlim(s_frame.valmin, s_frame.valmax)
        s_frame.set_val(0)
        redraw()

    def on_frame_change(_) -> None:
        state["frame"] = int(s_frame.val)
        redraw()

    def on_play(_) -> None:
        state["playing"] = not state["playing"]
        b_play.label.set_text("Pause" if state["playing"] else "Play")

    def on_timer() -> None:
        if state["playing"]:
            trial = state["trial"]
            next_frame = state["frame"] + 1
            if next_frame >= trial_steps[trial]:
                state["playing"] = False
                b_play.label.set_text("Play")
            else:
                s_frame.set_val(next_frame)

    s_trial.on_changed(on_trial_change)
    s_frame.on_changed(on_frame_change)
    b_play.on_clicked(on_play)
    timer = fig.canvas.new_timer(interval=120)
    timer.add_callback(on_timer)
    timer.start()

    redraw()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hill_fusion.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()

    arm = ThreeDofArm()
    arm.eval()

    if args.checkpoint:
        run_playback(arm, args.config, args.checkpoint, args.trials)
    else:
        run_manual(arm)


if __name__ == "__main__":
    main()
