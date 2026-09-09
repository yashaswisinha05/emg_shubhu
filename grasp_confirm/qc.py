"""The QC gate. Run this BEFORE you train anything.

    python -m grasp_confirm.qc --root data

It answers the only question that matters at 2am: is your AIR-vs-GRASP
contrast real, or did the protocol leak amplitude?

1. **Amplitude overlap.** AUC of mean log-RMS separating AIR from GRASP.
   0.5 means the classes are amplitude-matched (what you want). Above ~0.75
   means GRASP is simply louder and any classifier you train is an RMS
   detector wearing a neural network costume.

2. **Amplitude-only baseline.** Logistic regression on 4-channel log-RMS,
   held out by subject (or session). Above ~0.70 balanced accuracy means
   effort matching failed.

3. **Structure-only baseline.** The same classifier on gain-invariant
   features: trace-normalised channel covariance and 0.5-5 Hz envelope
   ripple. Multiplying every channel by a constant leaves these unchanged, so
   this classifier cannot be reading contraction level. **This is the number
   the night is for.**

4. **Per-condition breakdown.** AIR_LIGHT and AIR_HARD against GRASP
   separately. AIR_HARD is the adversarial pair -- those trials are LOUDER
   than the grasps and still have to come out as not-grasp. If the structure
   model holds up on AIR_HARD, amplitude is provably not what it is using.

A good outcome is check 1 near 0.5, check 2 near chance, check 3 well above
it, and check 4 showing no collapse on AIR_HARD.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import Window, load_dataset
from .features import amplitude_features, structure_features

RED, GREEN, YELLOW, BOLD, RESET = (
    "\033[31m",
    "\033[32m",
    "\033[33m",
    "\033[1m",
    "\033[0m",
)


def amp(window: Window) -> np.ndarray:
    return amplitude_features(window.x, window.fs)


def struct(window: Window) -> np.ndarray:
    return structure_features(window.x, window.fs)


def _groups(windows: list[Window]) -> tuple[np.ndarray, str]:
    if len({w.subject for w in windows}) >= 2:
        return np.array([w.subject for w in windows]), "leave-one-subject-out"
    if len({(w.subject, w.session) for w in windows}) >= 2:
        return (
            np.array([f"{w.subject}/{w.session}" for w in windows]),
            "leave-one-session-out (only one subject)",
        )
    return (
        np.array([str(w.trial) for w in windows]),
        "leave-one-trial-out (SINGLE SESSION -- weakest possible split)",
    )


def held_out_score(features: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """Balanced accuracy with one group held out at a time."""
    if len(np.unique(groups)) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    predictions = np.empty_like(labels)
    for train_idx, test_idx in LeaveOneGroupOut().split(features, labels, groups):
        if len(np.unique(labels[train_idx])) < 2:
            predictions[test_idx] = labels[train_idx][0]
            continue
        model.fit(features[train_idx], labels[train_idx])
        predictions[test_idx] = model.predict(features[test_idx])
    return float(balanced_accuracy_score(labels, predictions))


def verdict(value: float, good_below: float | None = None, good_above: float | None = None) -> str:
    if not np.isfinite(value):
        return f"{YELLOW}n/a{RESET}"
    if good_below is not None:
        return f"{GREEN}OK{RESET}" if value <= good_below else f"{RED}FAIL{RESET}"
    return f"{GREEN}OK{RESET}" if value >= (good_above or 0.0) else f"{YELLOW}WEAK{RESET}"


def run(root: Path, plot: Path | None = None) -> None:
    windows = load_dataset(root)
    if not windows:
        raise SystemExit(f"no sessions found under {root}")

    is_synthetic = any(w.synthetic for w in windows)
    if is_synthetic:
        print(
            f"{RED}{BOLD}SYNTHETIC DATA PRESENT.{RESET} Numbers below exercise the code "
            "path only; they are not a result and no verdict is issued.\n"
        )

    active = [w for w in windows if w.label in ("AIR", "GRASP")]
    counts = Counter(w.label for w in active)
    if counts["AIR"] < 5 or counts["GRASP"] < 5:
        raise SystemExit(
            f"need both classes; have AIR {counts['AIR']}, GRASP {counts['GRASP']}"
        )

    groups, split_name = _groups(active)
    labels = np.array([1 if w.label == "GRASP" else 0 for w in active])
    rest_n = sum(1 for w in windows if w.label == "REST")

    print(f"{BOLD}{len(active)} active windows{RESET}  AIR {counts['AIR']}  "
          f"GRASP {counts['GRASP']}  (+{rest_n} REST)")
    print(f"split: {split_name}")
    rates = sorted({w.fs for w in active})
    print(f"sample rate(s): {rates} Hz")
    if max(rates) < 1000.0:
        print(
            f"{YELLOW}sample rate below 1 kHz -- the 20-450 Hz band-pass and the raw-EMG "
            f"covariance features are not meaningful on an envelope stream.{RESET}"
        )
    print()

    amplitude = np.stack([amp(w) for w in active])
    structure = np.stack([struct(w) for w in active])

    mean_log_rms = amplitude.mean(axis=1)
    auc = float(roc_auc_score(labels, mean_log_rms))
    auc_sep = max(auc, 1.0 - auc)
    print(f"{BOLD}1. amplitude overlap{RESET}")
    print(f"   AUC of mean log-RMS, AIR vs GRASP : {auc_sep:.3f}   {verdict(auc_sep, good_below=0.75)}")
    print("   0.5 = perfectly amplitude-matched, 1.0 = GRASP is simply louder\n")

    amp_acc = held_out_score(amplitude, labels, groups)
    print(f"{BOLD}2. amplitude-only baseline (log-RMS){RESET}")
    print(f"   balanced accuracy : {amp_acc:.3f}   {verdict(amp_acc, good_below=0.70)}")
    print("   above 0.70 means effort matching failed and the task is trivial\n")

    str_acc = held_out_score(structure, labels, groups)
    both_acc = held_out_score(np.hstack([amplitude, structure]), labels, groups)
    print(f"{BOLD}3. structure-only baseline (gain-invariant){RESET}")
    print(f"   balanced accuracy : {str_acc:.3f}   {verdict(str_acc, good_above=0.65)}")
    print(f"   amplitude + structure : {both_acc:.3f}")
    print("   this is the number the night is for\n")

    print(f"{BOLD}4. per-condition breakdown{RESET}")
    grasp_windows = [w for w in active if w.label == "GRASP"]
    for condition in sorted({w.condition for w in active if w.label == "AIR"}):
        subset = [w for w in active if w.condition == condition] + grasp_windows
        if len(subset) < 12:
            continue
        sub_groups, _ = _groups(subset)
        sub_labels = np.array([1 if w.label == "GRASP" else 0 for w in subset])
        sub_amp = np.stack([amp(w) for w in subset])
        sub_str = np.stack([struct(w) for w in subset])
        rms_gap = sub_amp[sub_labels == 0].mean() - sub_amp[sub_labels == 1].mean()
        print(
            f"   {condition:12s} vs GRASP   amplitude {held_out_score(sub_amp, sub_labels, sub_groups):.3f}"
            f"   structure {held_out_score(sub_str, sub_labels, sub_groups):.3f}"
            f"   {'louder' if rms_gap > 0 else 'quieter'} than GRASP by {abs(rms_gap):.2f} log-units"
        )
    print("   AIR_HARD is the adversarial pair: louder than the grasps, still not-grasp\n")

    if is_synthetic:
        print(f"{RED}Synthetic data -- no verdict.{RESET}")
    elif auc_sep > 0.75 or amp_acc > 0.70:
        print(
            f"{RED}{BOLD}GATE FAILED.{RESET} GRASP and AIR are separable by loudness alone.\n"
            "Re-instruct: AIR_HARD must be squeezed harder than any grasp, and the cup\n"
            "trials held as gently as the object allows. Re-record a block and re-run\n"
            "this before training anything."
        )
    elif not np.isfinite(str_acc) or str_acc < 0.65:
        print(
            f"{YELLOW}{BOLD}GATE PASSED, SIGNAL WEAK.{RESET} Amplitude is properly matched but\n"
            "the gain-invariant features are near chance. Check that two sensors really\n"
            "are on the extensors -- without the flexor/extensor axis the co-activation\n"
            "feature has nothing to measure."
        )
    else:
        print(
            f"{GREEN}{BOLD}GATE PASSED.{RESET} Amplitude is matched and gain-invariant features\n"
            "separate the classes. Train the model."
        )

    if plot is not None:
        _plot(active, mean_log_rms, plot)
        print(f"\nwrote {plot}")


def _plot(windows: list[Window], mean_log_rms: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = sorted({w.subject for w in windows})
    fig, axes = plt.subplots(1, len(subjects), figsize=(5 * len(subjects), 4), squeeze=False)
    colours = {"AIR_LIGHT": "tab:green", "AIR_HARD": "tab:red"}
    for ax, subject in zip(axes[0], subjects):
        grasp = [v for v, w in zip(mean_log_rms, windows)
                 if w.subject == subject and w.label == "GRASP"]
        if grasp:
            ax.hist(grasp, bins=20, alpha=0.6, label="GRASP", color="tab:blue")
        for condition, colour in colours.items():
            values = [v for v, w in zip(mean_log_rms, windows)
                      if w.subject == subject and w.condition == condition]
            if values:
                ax.hist(values, bins=20, alpha=0.5, label=condition, color=colour)
        ax.set_title(subject)
        ax.set_xlabel("mean log RMS")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("windows")
    fig.suptitle("GRASP must sit BETWEEN AIR_LIGHT and AIR_HARD, not above both")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--plot", type=Path, default=Path("qc_amplitude.png"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    run(args.root, plot=None if args.no_plot else args.plot)


if __name__ == "__main__":
    main()
