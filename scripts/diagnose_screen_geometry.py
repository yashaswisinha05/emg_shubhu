#!/usr/bin/env python3
"""What would it actually take to hit 200 px? Answered from geometry, no model.

Six architectures have landed between 390 and 430 px on the pointing task.
Before building a seventh, this measures what the data itself permits, by
fitting the map from where the hand physically is to where the click lands.
No network, no training - least squares on a few thousand trials.

    python scripts/diagnose_screen_geometry.py \\
        --root "/media/.../emg_imu_vive" \\
        --config configs/tracked_grid_reach.yaml

Every fit below is scored on HELD-OUT trials and reported as mean Euclidean
pixel error - the same metric the models report, so the numbers are directly
comparable to 400 px and to the 200 px goal.

  1 CEILING. target_px ~ the 3-D touch position. This is a projection of a
    point onto a plane, so it should be near-exact. Whatever error remains
    is irreducible: tracker noise, finger-vs-tracker offset, click timing.
    NO model that works by locating the hand in space can beat this number,
    so it is the floor for the whole approach.

  2 SCALE. Pixels per centimetre, read off the same fit. This is the missing
    conversion between the two halves of this project: it turns the
    trajectory work's centimetres into the pointing work's pixels, and says
    how many centimetres of 3-D accuracy 200 px actually demands.

  3 THE ANCHOR TEST - the one that decides what to build next. Three fits:
      displacement only   onset-to-touch movement, no idea where it started
      start only          where the hand began, no idea where it went
      both                the full story
    A wearable can estimate displacement (measured: 27-41% better than
    baseline). It cannot know its own absolute position. So if displacement
    ALONE already explains the target, the ceiling is a prediction-accuracy
    problem and better models genuinely help. If it does not, no pointing
    model driven by wearables alone can reach 200 px, however good, and the
    effort belongs in supplying an anchor instead.

  4 PER SESSION vs GLOBAL. Fit 1 is repeated within each session. If the
    per-session fits are much tighter than the global one, the screen moved
    relative to the tracker between sessions - and a model tested on a
    held-out session is being asked to hit a target in a frame it has never
    observed, which is unanswerable rather than merely hard.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg_touch.config import load_config  # noqa: E402
from emg_touch.data.tracked_dataset import (  # noqa: E402
    discover_trials,
    preprocess_tracked_trial,
)


def fit_and_score(
    features: np.ndarray, targets_px: np.ndarray, seed: int = 0,
    train_fraction: float = 0.7,
) -> tuple[float, float, np.ndarray]:
    """Least squares with a bias, scored on held-out rows.

    Returns (held-out mean Euclidean pixel error, R^2, coefficients).
    Held out because an in-sample residual would flatter every fit here, and
    the whole point is to compare fits against each other honestly.
    """
    n = len(features)
    design = np.hstack([features, np.ones((n, 1))])
    order = np.random.default_rng(seed).permutation(n)
    cut = int(n * train_fraction)
    train, test = order[:cut], order[cut:]
    if len(test) < 8:
        return float("nan"), float("nan"), np.zeros(0)

    coefficients, *_ = np.linalg.lstsq(design[train], targets_px[train], rcond=None)
    predicted = design[test] @ coefficients
    residual = predicted - targets_px[test]
    error_px = float(np.linalg.norm(residual, axis=1).mean())

    spread = targets_px[test] - targets_px[train].mean(axis=0)
    r2 = float(1.0 - (residual**2).sum() / max((spread**2).sum(), 1e-9))
    return error_px, r2, coefficients


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/tracked_grid_reach.yaml")
    parser.add_argument("--limit", type=int, help="Sample this many trials per session.")
    args = parser.parse_args()

    config = load_config(args.config)
    sessions = discover_trials(args.root)
    print(f"{len(sessions)} session(s) found\n")

    starts, touches, targets, canvases, session_ids = [], [], [], [], []
    for session_index, (name, trials) in enumerate(sorted(sessions.items())):
        chosen = trials[: args.limit] if args.limit else trials
        for path in tqdm(chosen, desc=f"{name[:28]:28}", leave=False):
            data = preprocess_tracked_trial(path, config["data"])
            if data is None or "screen_target" not in data or "canvas" not in data:
                continue
            position = data["position"]
            onset = int(data["onset"])
            if len(position) < onset + 4:
                continue
            starts.append(position[onset])
            touches.append(position[-1])
            targets.append(data["screen_target"])
            canvases.append(data["canvas"])
            session_ids.append(session_index)

    if len(starts) < 50:
        print(f"only {len(starts)} usable trials - too few to fit", file=sys.stderr)
        sys.exit(2)

    starts = np.asarray(starts, dtype=np.float64)
    touches = np.asarray(touches, dtype=np.float64)
    canvas = np.asarray(canvases, dtype=np.float64)
    targets_px = np.asarray(targets, dtype=np.float64) * canvas
    session_ids = np.asarray(session_ids)
    displacement = touches - starts

    print(f"{len(starts)} usable trials across {len(set(session_ids))} sessions")
    print(f"canvas {canvas[0, 0]:.0f} x {canvas[0, 1]:.0f} px\n")

    # Guessing baselines, so every number below has something to beat.
    centre = np.full_like(targets_px, 0.5) * canvas
    mean_guess = np.tile(targets_px.mean(axis=0), (len(targets_px), 1))
    guess_mean_px = float(np.linalg.norm(mean_guess - targets_px, axis=1).mean())
    guess_centre_px = float(np.linalg.norm(centre - targets_px, axis=1).mean())

    print("=" * 74)
    print("1. CEILING - how well the 3-D touch point alone determines the click")
    print("=" * 74)
    ceiling_px, ceiling_r2, coefficients = fit_and_score(touches, targets_px)
    print(f"  touch position -> click :  {ceiling_px:7.1f} px   (R^2 {ceiling_r2:.4f})")
    print(f"  guessing the mean target:  {guess_mean_px:7.1f} px")
    print(f"  guessing screen centre  :  {guess_centre_px:7.1f} px")
    print()
    if ceiling_px < 200:
        print(f"  -> A perfect 3-D endpoint lands at {ceiling_px:.0f} px, so 200 px is")
        print("     geometrically REACHABLE. The gap is prediction accuracy.")
    else:
        print(f"  -> Pooled across sessions, even a PERFECT 3-D endpoint scores")
        print(f"     {ceiling_px:.0f} px. Do NOT read this as a physical limit until")
        print("     section 4: if the frame moved between sessions, this number is")
        print("     mostly the frame moving, not the geometry being loose.")

    # Every remaining fit is reported per session as well as globally. If the
    # tracker-to-screen frame moved between sessions, a pooled fit is
    # regressing pixels on coordinates that do not mean the same thing from
    # one session to the next, and every pooled number below inherits that.
    def per_session(build) -> tuple[float, float, list[np.ndarray]]:
        errors, jacobians = [], []
        for session in sorted(set(session_ids)):
            rows = session_ids == session
            if rows.sum() < 40:
                continue
            error, _, coefficient = fit_and_score(build(rows), targets_px[rows])
            if np.isfinite(error):
                errors.append(error)
                jacobians.append(coefficient[:3, :] if len(coefficient) > 3 else None)
        if not errors:
            return float("nan"), float("nan"), []
        return float(np.mean(errors)), float(np.std(errors)), jacobians

    ceiling_within, ceiling_spread, jacobians = per_session(lambda r: touches[r])

    print()
    print("=" * 74)
    print("2. SCALE - centimetres to pixels")
    print("=" * 74)
    # Taken from the per-session fits. A pooled Jacobian mixes frames and
    # collapses: on the real data it returned per-axis gains of 27.8 and 1.7
    # px/cm, a near-degenerate second axis that is an artefact of pooling
    # rather than a property of any screen.
    usable = [j for j in jacobians if j is not None]
    if usable:
        singular = np.mean(
            [np.linalg.svd(j, compute_uv=False) for j in usable], axis=0
        )
    else:
        singular = np.linalg.svd(coefficients[:3, :], compute_uv=False)
    px_per_cm = float(singular.mean() / 100.0)
    print(f"  {px_per_cm:.1f} px per cm  (per-axis gains {singular / 100.0} px/cm)")
    print(f"  200 px  =  {200.0 / max(px_per_cm, 1e-9):5.2f} cm of 3-D endpoint accuracy")
    print(f"  400 px  =  {400.0 / max(px_per_cm, 1e-9):5.2f} cm")
    print()
    print("  For comparison, already measured on the trajectory task:")
    for label, cm in (("short-horizon final (254 ms)", 8.44),
                      ("blind dead-reckoning, full reach", 19.12)):
        print(f"    {label:34}{cm:6.2f} cm  ->  {cm * px_per_cm:6.0f} px")

    print()
    print("=" * 74)
    print("3. THE ANCHOR TEST - is displacement enough on its own?")
    print("=" * 74)
    displacement_px, displacement_r2, _ = fit_and_score(displacement, targets_px)
    start_px, start_r2, _ = fit_and_score(starts, targets_px)
    both_px, both_r2, _ = fit_and_score(
        np.hstack([displacement, starts]), targets_px
    )
    displacement_within, _, _ = per_session(lambda r: displacement[r])
    start_within, _, _ = per_session(lambda r: starts[r])
    both_within, _, _ = per_session(
        lambda r: np.hstack([displacement[r], starts[r]])
    )
    print(f"  {'predictor':22}{'pooled px':>12}{'R^2':>8}{'within-session px':>20}")
    print("  " + "-" * 60)
    for label, error, r2, within in (
        ("displacement only", displacement_px, displacement_r2, displacement_within),
        ("start position only", start_px, start_r2, start_within),
        ("displacement + start", both_px, both_r2, both_within),
        ("touch position (ceiling)", ceiling_px, ceiling_r2, ceiling_within),
    ):
        print(f"  {label:22}{error:>12.1f}{r2:>8.4f}{within:>20.1f}")
    print()
    print("  The right-hand column is the one to read if the frame moved "
          "between\n  sessions - see section 4.")
    print()
    gap = displacement_px - both_px
    print(f"  Knowing where the reach STARTED is worth {gap:.0f} px "
          f"({displacement_px:.0f} -> {both_px:.0f}) pooled, and "
          f"{displacement_within - both_within:.0f} px within session.")
    if displacement_px < 200:
        print("  -> Displacement alone already determines the target well enough.")
        print("     A wearable CAN in principle reach 200 px; the ceiling we keep")
        print("     hitting is predictive accuracy, and better models pay off.")
    elif both_px < 200 <= displacement_px:
        print("  -> Displacement alone CANNOT get there, but displacement plus a")
        print("     start anchor can. This is the decisive result: the missing")
        print("     ingredient is an anchor, not a better encoder. Every model so")
        print("     far has been asked an under-determined question.")
    else:
        print("  -> Neither reaches 200 px even with perfect knowledge, which puts")
        print("     the limit in the click/tracker relationship itself.")

    print()
    print("=" * 74)
    print("4. PER-SESSION vs GLOBAL - does the screen move between sessions?")
    print("=" * 74)
    if np.isfinite(ceiling_within):
        within = ceiling_within
        print(f"  global fit (one frame for all sessions) : {ceiling_px:7.1f} px")
        print(f"  per-session fits, averaged              : {within:7.1f} px")
        print(f"  spread across sessions                  : {ceiling_spread:7.1f} px")
        print()
        if ceiling_px > within * 1.5:
            print("  -> Per-session fits are much tighter, so the tracker-to-screen")
            print("     frame MOVED between sessions. A model tested on a held-out")
            print("     session must infer a frame it has never seen - that part of")
            print("     the error is unanswerable, not merely hard, and it caps every")
            print("     held-out-session result reported so far.")
        else:
            print("  -> The frame is stable across sessions, so held-out-session")
            print("     testing is fair and this is not what has been limiting us.")

    start_spread_cm = float(np.linalg.norm(starts - starts.mean(axis=0), axis=1).mean() * 100)
    print()
    print(f"  (reach start positions vary by {start_spread_cm:.1f} cm on average, "
          f"~{start_spread_cm * px_per_cm:.0f} px worth)")


if __name__ == "__main__":
    main()
