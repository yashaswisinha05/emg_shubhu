#!/usr/bin/env python3
"""Diagnostic only: does EMG separate open/close by spatial pattern or by gain?

No training, no model, nothing written back to the dataset.

Context. Every trial in this dataset carries a single constant gripper_state,
and the two classes live in separate recording sessions, so session identity
predicts the label perfectly. IMU-only reaching macro F1 1.000 already proves
a session-constant signature exists -- a forearm IMU cannot see hand aperture,
so it has nothing else to read. This script asks a narrower question about the
EMG: does the class separation survive removing each trial's overall gain?

Three trial-level feature sets, same classifier, same folds. Each trial is one
sample (n = number of trials), because the label never changes within a trial.

  level     2 features -- mean short/long trailing RMS pooled over all 4
            sensors. "How loud was the EMG", nothing else. Electrode
            impedance, skin prep and amplifier gain all move this, and all
            three differ between sessions.
  absolute  8 features -- per-sensor, per-timescale mean RMS. Level + pattern.
  pattern   8 features -- each sensor's share of its timescale's total, so
            overall gain divides out and only the distribution across the 4
            sensors survives. A sustained grip has a specific signature here
            (flexors elevated relative to extensors); a pure gain or impedance
            offset scales all sensors together and largely does not.

Reading the result:
  level high, pattern near chance  -> separation is overall gain. Artifact-flavoured.
  pattern stays high               -> real spatial structure across sensors.

Two further checks probe the other artifact that survives gain removal: an
electrode state that changes slowly over a recording (drying, gel settling,
sweat, fatigue). Both work within a session, where the class is constant, so
neither can be satisfied by the between-session difference.

  drift           each share regressed against the trial's position in its own
                  session, reported as the movement from first trial to last.
                  Compare it against the between-class gap: comparable size
                  means the shares wander on their own.
  temporal split  train on one half of every session in recording order, test
                  on the other half. A drop means the signature moves across a
                  session instead of being a stable property of the condition.

CAVEAT, and it matters. A rotated or shifted armband placement between the two
sessions ALSO changes channel ratios, because each sensor then sits over a
different muscle, and no within-session check can see that -- placement is
fixed for the whole session. So "pattern separates, and is stable within each
session" is evidence for physiology, not proof of it. The decisive test remains
one new session with both conditions interleaved, trained and tested within
that session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.data.gripper_state import add_gripper_state
from emg_touch.data.reach_grasp import preprocess


def trial_features(trial):
    """Mean trailing RMS per EMG channel, over that channel's valid samples.

    preprocess packs EMG as 8 channels: sensors 0-3 at the 20 ms RMS scale,
    then the same 4 sensors at the 50 ms scale.
    """
    value, valid = trial["emg"], trial["emg_valid"].astype(bool)
    count = valid.sum(0)
    if (count < 10).any():
        raise ValueError("a channel has fewer than 10 valid samples")
    mean_rms = (value * valid).sum(0) / count
    if not np.isfinite(mean_rms).all() or (mean_rms <= 0).any():
        raise ValueError("nonpositive or nonfinite mean RMS")
    short, long = mean_rms[:4], mean_rms[4:]
    level = np.log(np.array([short.mean(), long.mean()]))
    absolute = np.log(mean_rms)
    pattern = np.concatenate([short / short.sum(), long / long.sum()])
    return {"level": level, "absolute": absolute, "pattern": pattern}


def within_session(sessions, numbers, session):
    """Mask of one session's trials, plus each trial's position within it."""
    mask = sessions == session
    order = np.argsort(numbers[mask], kind="stable")
    position = np.empty(int(mask.sum()), dtype=float)
    position[order] = np.arange(mask.sum(), dtype=float)
    return mask, position


def drift(values, position):
    """Total movement of a feature from the session's first trial to its last."""
    if len(position) < 3 or position.max() == position.min():
        return 0., 0.
    slope = float(np.polyfit(position, values, 1)[0])
    return slope * (position.max() - position.min()), float(np.corrcoef(position, values)[0, 1])


def temporal_split(features, labels, sessions, numbers, test_late):
    """Train on one half of every session in recording order, test on the other.

    Both classes stay present on both sides because the halves are taken
    within each session, so this isolates drift across a session rather than
    re-testing the between-session difference.
    """
    train = np.zeros(len(labels), dtype=bool)
    for session in np.unique(sessions):
        mask, position = within_session(sessions, numbers, session)
        early = position < (position.max() + 1) / 2
        train[np.flatnonzero(mask)[early if test_late else ~early]] = True
    if min(len(np.unique(labels[train])), len(np.unique(labels[~train]))) < 2:
        return None
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(features[train], labels[train])
    predicted = model.predict(features[~train])
    return {"accuracy": float(accuracy_score(labels[~train], predicted)),
            "macro_f1": float(f1_score(labels[~train], predicted, average="macro")),
            "train_trials": int(train.sum()), "test_trials": int((~train).sum())}


def score(features, labels, seed, shuffle_labels=False):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    splits = min(5, int(np.bincount(labels).min()))
    folds = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    target = labels
    if shuffle_labels:
        target = np.random.default_rng(seed).permutation(labels)
    accuracy = cross_val_score(model, features, target, cv=folds, scoring="accuracy")
    f1 = cross_val_score(model, features, target, cv=folds, scoring="f1_macro")
    return {"accuracy_mean": float(accuracy.mean()), "accuracy_std": float(accuracy.std()),
            "macro_f1_mean": float(f1.mean()), "macro_f1_std": float(f1.std())}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--raw-rate-hz", type=float, default=1259.4)
    parser.add_argument("--event-origin", choices=["auto", "start"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    settings = {"raw_rate_hz": args.raw_rate_hz, "rate_hz": 100., "gap_s": .02,
                "event_origin": args.event_origin, "event_pulse_s": .1,
                "require_events": False}
    discovered = {}
    for root in args.root:
        for path in Path(root).rglob("trial_*.csv"):
            discovered.setdefault(path, root)
    if not discovered:
        raise ValueError("no files matched 'trial_*.csv' under: " + ", ".join(args.root))

    rows, labels, sessions, numbers = [], [], [], []
    rejected, hashes, transitions = {}, {}, 0
    for path in sorted(discovered):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            rejected[str(path)] = "byte-identical duplicate of " + hashes[digest]
            continue
        hashes[digest] = str(path)
        try:
            trial = add_gripper_state(path, preprocess(path, settings), settings["gap_s"])
            state = trial["gripper_state"][trial["gripper_state_valid"]]
            if not len(state):
                raise ValueError("no valid gripper_state samples")
            if len(np.unique(state)) > 1:
                transitions += 1
            rows.append(trial_features(trial))
            labels.append(int(np.bincount(state, minlength=2).argmax()))
            sessions.append(discovered[path])
            match = re.search(r"(\d+)", path.stem)
            numbers.append(int(match.group(1)) if match else -1)
        except (ValueError, KeyError) as error:
            rejected[str(path)] = str(error)

    labels = np.asarray(labels)
    sessions = np.asarray(sessions)
    numbers = np.asarray(numbers)
    if len(labels) < 20 or len(np.unique(labels)) < 2:
        raise ValueError(f"need >=20 trials spanning both classes, got {len(labels)} "
                         f"({len(rejected)} rejected)")

    counts = np.bincount(labels, minlength=2)
    print(f"trials: {len(labels)} usable ({counts[0]} open, {counts[1]} close), "
          f"{len(rejected)} rejected, {transitions} containing an open/close transition")
    print(f"chance accuracy (majority class): {counts.max() / counts.sum():.3f}\n")

    results = {"trials": int(len(labels)), "open": int(counts[0]), "close": int(counts[1]),
               "trials_with_transition": transitions, "rejected": len(rejected),
               "majority_class_accuracy": float(counts.max() / counts.sum())}
    sets = {name: np.stack([row[name] for row in rows]) for name in ("level", "absolute", "pattern")}
    print(f"{'feature set':<24}{'accuracy':>18}{'macro F1':>18}")
    for name, matrix in sets.items():
        results[name] = score(matrix, labels, args.seed)
        results[name]["features"] = int(matrix.shape[1])
        r = results[name]
        print(f"{name:<24}{r['accuracy_mean']:>10.3f} +-{r['accuracy_std']:<6.3f}"
              f"{r['macro_f1_mean']:>10.3f} +-{r['macro_f1_std']:<6.3f}")
    results["pattern_shuffled_control"] = score(sets["pattern"], labels, args.seed,
                                                shuffle_labels=True)
    r = results["pattern_shuffled_control"]
    print(f"{'pattern (shuffled)':<24}{r['accuracy_mean']:>10.3f} +-{r['accuracy_std']:<6.3f}"
          f"{r['macro_f1_mean']:>10.3f} +-{r['macro_f1_std']:<6.3f}")

    pattern = sets["pattern"]
    print("\nper-sensor share of its timescale total (mean over trials):")
    print(f"{'':<16}{'sensor 0':>10}{'sensor 1':>10}{'sensor 2':>10}{'sensor 3':>10}")
    shares = {}
    for scale, columns in (("short", slice(0, 4)), ("long", slice(4, 8))):
        for name, target in (("open", 0), ("close", 1)):
            mean = pattern[labels == target, columns].mean(0)
            shares[f"{scale}_{name}"] = mean.tolist()
            print(f"{scale + ' ' + name:<16}" + "".join(f"{v:>10.4f}" for v in mean))
        delta = np.asarray(shares[f"{scale}_close"]) - np.asarray(shares[f"{scale}_open"])
        shares[f"{scale}_delta"] = delta.tolist()
        print(f"{scale + ' delta':<16}" + "".join(f"{v:>+10.4f}" for v in delta))
    results["pattern_shares"] = shares

    print("\nwithin-session drift of the same shares, against position in the session.")
    print("drift = movement from the session's first trial to its last, so compare it")
    print("against the between-session delta printed above: drift of similar size means")
    print("the shares wander on their own and the class gap need not be physiological.")
    drifts = {}
    for scale, columns in (("short", slice(0, 4)), ("long", slice(4, 8))):
        print(f"{'':<22}{'sensor 0':>10}{'sensor 1':>10}{'sensor 2':>10}{'sensor 3':>10}")
        for session in np.unique(sessions):
            mask, position = within_session(sessions, numbers, session)
            block = pattern[mask, columns]
            values = [drift(block[:, sensor], position) for sensor in range(4)]
            name = Path(session).name or session
            drifts[f"{scale}_{name}"] = {"drift": [v[0] for v in values],
                                         "pearson_r": [v[1] for v in values]}
            print(f"{scale + ' ' + name:<22}" + "".join(f"{v:>+10.4f}" for v, _ in values))
            print(f"{'  pearson r':<22}" + "".join(f"{r:>+10.3f}" for _, r in values))
        print(f"{scale + ' class gap':<22}"
              + "".join(f"{v:>+10.4f}" for v in shares[f"{scale}_delta"]))
    results["within_session_drift"] = drifts

    print("\ntemporal split: train on one half of every session in recording order,")
    print("test on the other half. Both classes are present on both sides, so a drop")
    print("here means the signature moves across a session rather than being stable.")
    print(f"{'feature set':<24}{'early -> late':>18}{'late -> early':>18}")
    splits = {}
    for name, matrix in sets.items():
        late = temporal_split(matrix, labels, sessions, numbers, test_late=True)
        early = temporal_split(matrix, labels, sessions, numbers, test_late=False)
        splits[name] = {"train_early_test_late": late, "train_late_test_early": early}
        print(f"{name:<24}{late['accuracy']:>18.3f}{early['accuracy']:>18.3f}"
              if late and early else f"{name:<24}{'unavailable':>36}")
    results["temporal_split"] = splits

    print("\nlog mean RMS pooled over sensors (level features):")
    level = sets["level"]
    for scale, column in (("short", 0), ("long", 1)):
        open_mean, close_mean = level[labels == 0, column].mean(), level[labels == 1, column].mean()
        results[f"level_{scale}"] = {"open": float(open_mean), "close": float(close_mean)}
        print(f"  {scale:<6} open {open_mean:+.3f}   close {close_mean:+.3f}   "
              f"delta {close_mean - open_mean:+.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
