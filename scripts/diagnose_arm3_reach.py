#!/usr/bin/env python3
"""Headless check that the 3-DOF arm is behaving like an arm.

Answers, on a real trained checkpoint and real trials, the question the
interactive viewer answers by eye - but without needing a display, so it runs
over SSH on a training box:

  - do the joints actually move, or sit pinned at their limits?
  - does the prescribed shoulder really track the measured IMU displacement?
  - does the elbow differ from trial to trial, i.e. is EMG driving it at all,
    or is it replaying one canned motion regardless of input?
  - is the hand inside the arm's reachable workspace, with link lengths
    preserved (a check on the forward kinematics itself)?
  - does the endpoint move toward the click target over the trial - not
    just in normalised screen units, but in real metres against an actual
    screen plane placed at the measured shoulder-to-screen distance?

Writes one figure per trial - a 3-D view of the arm reaching (or not)
toward a drawn screen plane at the real measured distance, with the click
target placed on it as an actual 3-D point and a dashed line showing the
remaining gap; joint angles over time; and screen-space predictions
against the target - plus a summary scatter, and prints a numeric verdict.
The printed report is the primary output; the figures are supporting
evidence to scp back and look at.

Usage:
  python scripts/diagnose_arm3_reach.py \
    --config configs/hill_fusion.yaml \
    --checkpoint runs/physics3_imu_driven/grid_fusion_physics3/best.pt \
    --device cuda --trials 6 --output-dir evaluation/arm3_diagnosis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot: this box has no display

import matplotlib.pyplot as plt
import numpy as np
import torch

from emg_touch.config import load_config
from emg_touch.data.grid_trajectory import build_grid_trajectory_loaders
from emg_touch.models.grid_point import build_grid_model
from emg_touch.utils import choose_device, move_batch_to_device

JOINTS = ("q1 shoulder-z", "q2 shoulder-y", "q3 elbow")
LOWER = np.array([-1.8, -1.8, 0.0])
UPPER = np.array([1.8, 1.8, 2.8])


def chain_points(arm, angles: torch.Tensor) -> np.ndarray:
    """Shoulder/elbow/hand positions (N, 3, 3) straight from the arm's own FK."""
    _, t12, _ = arm._fk(angles)
    elbow_home = torch.tensor(
        [float(arm.link_length[0]), 0.0, 0.0, 1.0],
        device=angles.device,
        dtype=angles.dtype,
    )
    elbow = (t12 @ elbow_home)[:, :3]
    hand = arm.endpoint(angles)
    shoulder = torch.zeros_like(hand)
    return torch.stack([shoulder, elbow, hand], dim=1).detach().cpu().numpy()


def screen_point_3d(
    normalised_xy: np.ndarray, distance: float, width: float, height: float
) -> np.ndarray:
    """Map a normalised [0,1]x[0,1] screen coordinate (y=0 at top) to a 3-D
    point on a screen plane placed `distance` metres along the arm's home
    reach direction (+x), centred in front of the shoulder. Screen width and
    height are plotting assumptions, not measurements - the geometry that
    *is* measured (arm reach vs. screen distance) is what this whole
    visualiser exists to check; exact panel size only affects where on the
    drawn rectangle the target dot sits, not whether the arm's reach lines
    up with the rectangle itself.
    """
    x = np.full(normalised_xy.shape[:-1], distance)
    y = (normalised_xy[..., 0] - 0.5) * width
    z = (0.5 - normalised_xy[..., 1]) * height
    return np.stack([x, y, z], axis=-1)


def draw_screen_plane(ax, distance: float, width: float, height: float) -> None:
    corners_y = np.array([-1, 1, 1, -1, -1]) * (width / 2)
    corners_z = np.array([-1, -1, 1, 1, -1]) * (height / 2)
    ax.plot(np.full(5, distance), corners_y, corners_z, color="#888780", lw=1.2)
    xx, yy = np.meshgrid([distance, distance], [-width / 2, width / 2])
    zz = np.array([[-height / 2, -height / 2], [height / 2, height / 2]])
    ax.plot_surface(xx, yy, zz, color="#B4B2A9", alpha=0.15, shade=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hill_fusion.yaml")
    parser.add_argument("--kind", default="grid_fusion_physics3")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--output-dir", default="evaluation/arm3_diagnosis")
    parser.add_argument(
        "--screen-distance", type=float, default=1.10,
        help="Measured distance from shoulder to screen, metres.",
    )
    parser.add_argument(
        "--screen-width", type=float, default=0.34,
        help="Screen width, metres - not measured, a plotting default (adjust to your rig).",
    )
    parser.add_argument(
        "--screen-height", type=float, default=0.20,
        help="Screen height, metres - not measured, a plotting default (adjust to your rig).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device(args.device)
    model = build_grid_model(args.kind, config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, val_loader, _ = build_grid_trajectory_loaders(
        config, config["paths"]["split_file"], config["paths"]["scaler"]
    )
    batch = move_batch_to_device(next(iter(val_loader)), device)
    with torch.no_grad():
        outputs = model(batch)

    arm = model.physics.arm
    decimation = int(config.get("physics", {}).get("decimation", 4))
    trajectory = outputs["physics_trajectory"].detach()          # (B, K, 3)
    lengths = batch["lengths"].detach().cpu().numpy()
    target = batch["target"].detach().cpu().numpy()
    physics_pred = outputs["physics_prediction"].detach().cpu().numpy()
    fusion_pred = outputs["fusion_prediction"].detach().cpu().numpy()
    blend = outputs["physics_blend"].detach().cpu().numpy()
    measured = outputs.get("physics_shoulder_measured")
    if measured is not None:
        measured = measured.detach()

    batch_size = trajectory.size(0)
    steps_for = lambda i: max(1, int((lengths[i] + decimation - 1) // decimation))

    print("=" * 72)
    print(f"checkpoint: {args.checkpoint}  (epoch {checkpoint.get('epoch', '?')})")
    print(f"imu_driven_shoulder={model.physics.imu_driven_shoulder}  "
          f"gravity_compensation={model.physics.gravity_compensation}  "
          f"torque_scale={model.physics.torque_scale}")
    print("=" * 72)

    # ---- 1. joint motion and limit saturation -------------------------------
    print("\n[1] Joint motion (per trial, over its own valid length)")
    ranges = np.zeros((batch_size, 3))
    at_limit = np.zeros((batch_size, 3))
    for i in range(batch_size):
        k = steps_for(i)
        q = trajectory[i, :k].cpu().numpy()
        ranges[i] = q.max(axis=0) - q.min(axis=0)
        near_low = np.abs(q - LOWER) < 1e-6
        near_high = np.abs(q - UPPER) < 1e-6
        at_limit[i] = (near_low | near_high).mean(axis=0)
    for j, name in enumerate(JOINTS):
        print(f"  {name:<16} travel: mean={ranges[:,j].mean():.3f} rad "
              f"({np.degrees(ranges[:,j].mean()):5.1f} deg)  "
              f"min={ranges[:,j].min():.3f}  max={ranges[:,j].max():.3f}   "
              f"| time at a joint limit: {at_limit[:,j].mean()*100:5.1f}%")
    dead = [JOINTS[j] for j in range(3) if ranges[:, j].mean() < 1e-3]
    print("  VERDICT:", "all three joints move" if not dead else f"NOT MOVING: {dead}")
    # The elbow is the one joint physics exists to infer - the shoulder is
    # handed to it by the IMU. An elbow that barely moves next to a sweeping
    # shoulder means the branch is riding the measurement and contributing
    # almost nothing of its own, which no endpoint number would reveal.
    shoulder_travel = ranges[:, 0:2].max(axis=1).mean()
    elbow_travel = ranges[:, 2].mean()
    print(f"  elbow travel vs shoulder travel: {elbow_travel:.3f} / {shoulder_travel:.3f} rad "
          f"= {elbow_travel / max(shoulder_travel, 1e-9):.2f}x")
    if elbow_travel < 0.05:
        print("  WARNING: elbow is effectively static - physics is riding the "
              "prescribed shoulder and inferring almost nothing from EMG.")

    # ---- 2. does the prescribed shoulder track the measurement? -------------
    if measured is not None:
        print("\n[2] Shoulder tracking (physics trajectory vs measured IMU displacement)")
        errs, cors = [], []
        for i in range(batch_size):
            k = steps_for(i)
            got = trajectory[i, :k, 0:2].cpu().numpy()
            want = measured[i, :k].cpu().numpy()
            errs.append(np.abs(got - want).max())
            for ax in range(2):
                if got[:, ax].std() > 1e-9 and want[:, ax].std() > 1e-9:
                    cors.append(np.corrcoef(got[:, ax], want[:, ax])[0, 1])
        print(f"  max |physics shoulder - measured| = {max(errs):.2e} rad")
        print(f"  correlation (per axis, all trials): mean={np.mean(cors):.4f} "
              f"min={np.min(cors):.4f}")
        print("  VERDICT:", "shoulder is following the measurement"
              if max(errs) < 1e-4 else "MISMATCH - shoulder is not tracking IMU")
    else:
        print("\n[2] Shoulder tracking: skipped (imu_driven_shoulder is off)")

    # ---- 3. is the elbow input-driven, or one canned motion? ----------------
    print("\n[3] Elbow is actually driven by input (not a fixed replay)")
    final_elbow = trajectory[:, -1, 2].cpu().numpy()
    curves = np.stack([
        np.interp(np.linspace(0, 1, 40),
                  np.linspace(0, 1, steps_for(i)),
                  trajectory[i, :steps_for(i), 2].cpu().numpy())
        for i in range(batch_size)
    ])
    pairwise = [
        np.abs(curves[a] - curves[b]).mean()
        for a in range(batch_size) for b in range(a + 1, batch_size)
    ]
    print(f"  final elbow across trials: mean={final_elbow.mean():.3f} "
          f"std={final_elbow.std():.3f} min={final_elbow.min():.3f} max={final_elbow.max():.3f}")
    print(f"  mean pairwise difference between trials' elbow curves: {np.mean(pairwise):.4f} rad")
    print("  VERDICT:", "elbow differs per trial (input matters)"
          if np.mean(pairwise) > 1e-3 else "IDENTICAL across trials - elbow ignores input")

    # ---- 4. forward kinematics sanity --------------------------------------
    print("\n[4] Forward kinematics sanity")
    flat = trajectory.reshape(-1, 3)
    pts = chain_points(arm, flat)
    seg1 = np.linalg.norm(pts[:, 1] - pts[:, 0], axis=-1)
    seg2 = np.linalg.norm(pts[:, 2] - pts[:, 1], axis=-1)
    l1, l2 = float(arm.link_length[0]), float(arm.link_length[1])
    reach = np.linalg.norm(pts[:, 2], axis=-1)
    print(f"  upper arm length: want {l1:.4f}  got {seg1.min():.4f}-{seg1.max():.4f} m")
    print(f"  forearm length:   want {l2:.4f}  got {seg2.min():.4f}-{seg2.max():.4f} m")
    print(f"  hand distance from shoulder: {reach.min():.3f}-{reach.max():.3f} m "
          f"(max possible {l1+l2:.3f})")
    ok = (abs(seg1 - l1).max() < 1e-4 and abs(seg2 - l2).max() < 1e-4
          and reach.max() <= l1 + l2 + 1e-4)
    print("  VERDICT:", "links rigid and hand inside workspace" if ok
          else "GEOMETRY ERROR - link lengths or workspace violated")

    # ---- 5. does the endpoint approach the target? -------------------------
    print("\n[5] Screen-space accuracy (normalised units, 0-1 across the canvas)")
    d_phys = np.linalg.norm(physics_pred - target, axis=-1)
    d_fuse = np.linalg.norm(fusion_pred - target, axis=-1)
    print(f"  physics endpoint -> target: mean={d_phys.mean():.4f}")
    print(f"  fusion  endpoint -> target: mean={d_fuse.mean():.4f}")
    print(f"  physics_blend: mean={blend.mean():.5f} min={blend.min():.5f} max={blend.max():.5f}")
    closer = (d_phys < d_fuse).mean()
    print(f"  physics closer than fusion on {closer*100:.0f}% of trials")

    # Same accuracy question, but in real metres against a screen placed at
    # the measured distance - normalised units above can't say whether the
    # arm's hand is anywhere near the screen at all, only how the affine's
    # already-fitted output compares; this measures the hand's own physical
    # position against where the target actually is in space.
    final_hand = torch.stack(
        [trajectory[b, steps_for(b) - 1] for b in range(batch_size)]
    )
    final_hand_xyz = arm.endpoint(final_hand).detach().cpu().numpy()
    target_xyz = screen_point_3d(target, args.screen_distance, args.screen_width, args.screen_height)
    reach_gap = np.linalg.norm(final_hand_xyz - target_xyz, axis=-1)
    print(f"  hand -> screen-plane target, in real metres: mean={reach_gap.mean():.3f} "
          f"min={reach_gap.min():.3f} max={reach_gap.max():.3f} "
          f"(screen assumed {args.screen_width:.2f}x{args.screen_height:.2f} m "
          f"at {args.screen_distance:.2f} m)")

    # ---- 6. is a folded elbow a real optimum, or just undertrained? --------
    # If the elbow has settled near one end of its range with almost no
    # per-trial variation, that alone does not say whether it is a genuine
    # local optimum (more training would not fix it) or simply undertrained
    # (more training likely would). Bias the loaded model's own torque head
    # toward extension/flexion directly and see whether physics gets closer
    # to target - the actual model, the actual weights, no synthetic stand-in.
    print("\n[6] Elbow bias sweep on this checkpoint (is folding actually helping?)")
    torque_head = getattr(model.physics, "torque_head", None)
    if torque_head is None:
        print("  skipped: this branch has no torque_head (Hill muscle model)")
    else:
        final_layer = torque_head.net[-1]
        original_bias = final_layer.bias.detach().clone()
        results = []
        for delta in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
            with torch.no_grad():
                final_layer.bias.copy_(original_bias)
                final_layer.bias[2] += delta  # negative = push toward extension
                out = model(batch)
            d = np.linalg.norm(
                out["physics_prediction"].detach().cpu().numpy() - target, axis=-1
            ).mean()
            elbow_now = out["physics_angles"][:, 2].mean().item()
            results.append((delta, elbow_now, d))
            print(f"  bias delta={delta:+.1f} -> elbow mean={elbow_now:.3f} rad  "
                  f"physics->target={d:.4f}")
        with torch.no_grad():
            final_layer.bias.copy_(original_bias)
        best = min(results, key=lambda r: r[2])
        baseline = next(r for r in results if r[0] == 0.0)
        if best[0] == 0.0:
            print("  VERDICT: current elbow policy is already the best of those tried "
                  "- consistent with undertrained rather than a bad optimum.")
        elif best[2] < baseline[2] * 0.97:
            direction = "more extension (toward 0)" if best[0] < 0 else "more flexion (toward 2.8)"
            print(f"  VERDICT: {direction} measurably reduces error ({best[2]:.4f} vs "
                  f"{baseline[2]:.4f} at the trained bias) - the trained elbow angle is "
                  "leaving accuracy on the table, not just undertrained on other things.")
        else:
            print("  VERDICT: no bias tried beats the trained setting by a meaningful "
                  "margin - the elbow's current position is close to locally optimal "
                  "given everything else in the model as it stands.")

    # ---- figures -----------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shown = min(args.trials, batch_size)
    for i in range(shown):
        k = steps_for(i)
        q = trajectory[i, :k]
        fig = plt.figure(figsize=(15, 4.2))

        ax = fig.add_subplot(131, projection="3d")
        draw_screen_plane(ax, args.screen_distance, args.screen_width, args.screen_height)
        picks = np.linspace(0, k - 1, min(6, k)).astype(int)
        pts_i = chain_points(arm, q[picks])
        for n, p in enumerate(pts_i):
            shade = 0.78 - 0.68 * (n / max(1, len(pts_i) - 1))
            ax.plot(p[:, 0], p[:, 1], p[:, 2], "-o", color=str(shade),
                    lw=3, ms=4, label="start" if n == 0 else ("touch" if n == len(pts_i) - 1 else None))
        hand_final = pts_i[-1, 2]
        target_3d = screen_point_3d(
            target[i], args.screen_distance, args.screen_width, args.screen_height
        )
        ax.scatter(*target_3d, s=70, c="#D85A30", marker="x", label="click target", zorder=5)
        ax.plot(
            [hand_final[0], target_3d[0]], [hand_final[1], target_3d[1]],
            [hand_final[2], target_3d[2]], "--", color="#D85A30", lw=1, alpha=0.7,
        )
        x_lo, x_hi = -0.05, args.screen_distance * 1.12
        yz_span = max(float(arm.link_length.sum()), args.screen_width / 2, args.screen_height / 2)
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(-yz_span, yz_span); ax.set_zlim(-yz_span, yz_span)
        ax.view_init(elev=18, azim=-58)
        ax.set_xlabel("reach (m)"); ax.set_ylabel("lateral (m)"); ax.set_zlabel("vertical (m)")
        gap = float(np.linalg.norm(hand_final - target_3d))
        ax.set_title(f"arm vs. screen (dark=touch, gap to target={gap:.2f} m)")
        ax.legend(fontsize=7, loc="upper left")

        ax2 = fig.add_subplot(132)
        t = np.arange(k)
        for j, name in enumerate(JOINTS):
            ax2.plot(t, q[:, j].cpu().numpy(), label=name)
        if measured is not None:
            for ax_i, style in enumerate(("--", ":")):
                ax2.plot(t, measured[i, :k, ax_i].cpu().numpy(), style,
                         color="k", lw=1, alpha=0.6,
                         label="measured shoulder" if ax_i == 0 else None)
        ax2.axhline(0.0, color="0.85", lw=0.8)
        ax2.axhline(2.8, color="0.85", lw=0.8)
        ax2.set_xlabel("decimated step"); ax2.set_ylabel("angle (rad)")
        ax2.set_title("joint angles"); ax2.legend(fontsize=7)

        ax3 = fig.add_subplot(133)
        ax3.scatter(*target[i], s=120, c="#D85A30", label="click target")
        ax3.scatter(*fusion_pred[i], s=90, c="#378ADD", label="fusion")
        ax3.scatter(*physics_pred[i], s=90, c="#0F6E56", label="physics")
        ax3.set_xlim(0, 1); ax3.set_ylim(1, 0); ax3.set_aspect("equal")
        ax3.set_title(f"screen (blend={blend[i]:.4f})"); ax3.legend(fontsize=7)

        fig.suptitle(f"trial {batch['trial_id'][i]}", fontsize=9)
        fig.tight_layout()
        path = out / f"trial_{i:02d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    for name, pred, colour in (
        ("fusion", fusion_pred, "#378ADD"), ("physics", physics_pred, "#0F6E56")
    ):
        ax.scatter(pred[:, 0], pred[:, 1], s=26, c=colour, label=name, alpha=0.8)
    ax.scatter(target[:, 0], target[:, 1], s=42, c="#D85A30", marker="x", label="target")
    for i in range(batch_size):
        ax.plot([physics_pred[i, 0], target[i, 0]], [physics_pred[i, 1], target[i, 1]],
                color="0.8", lw=0.6, zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
    ax.set_title("all trials: predictions vs targets"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "summary_screen.png", dpi=120)
    plt.close(fig)

    print(f"\nWrote {shown} trial figures + summary_screen.png to {out.resolve()}")


if __name__ == "__main__":
    main()
