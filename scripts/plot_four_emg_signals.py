#!/usr/bin/env python3
"""Plot any four EMG channels from a recorded trial CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def emg_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns whose names contain EMG, preserving CSV order."""
    return [name for name in frame.columns if "emg" in name.lower()]


def time_axis(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    for name in ("time_s", "time_perf_counter"):
        if name in frame:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            finite = np.isfinite(values)
            if finite.any():
                origin = values[finite][0]
                return values - origin, "Time (s)"
    return np.arange(len(frame), dtype=float), "Sample"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-csv", type=Path, required=True)
    parser.add_argument("--columns", nargs=4,
                        help="Exact four column names; default: first four EMG columns")
    parser.add_argument("--start-s", type=float)
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--normalize", action="store_true",
                        help="Robustly center and scale each displayed channel")
    parser.add_argument("--output", type=Path,
                        help="Default: <trial-name>_four_emg.png beside the CSV")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        parser.error("matplotlib is required; install it with: pip install matplotlib")

    frame = pd.read_csv(args.trial_csv)
    columns = args.columns or emg_columns(frame)[:4]
    if len(columns) != 4:
        parser.error(f"found only {len(columns)} EMG column(s); pass --columns explicitly")
    missing = [name for name in columns if name not in frame]
    if missing:
        parser.error("missing columns: " + ", ".join(missing))

    time, xlabel = time_axis(frame)
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    keep = np.isfinite(time)
    if args.start_s is not None:
        keep &= time >= args.start_s
    if args.end_s is not None:
        keep &= time <= args.end_s
    time, values = time[keep], values[keep]
    if not len(time):
        parser.error("selected time range contains no samples")
    order = np.argsort(time, kind="stable")
    time, values = time[order], values[order]
    unique = np.r_[True, np.diff(time) > 0]
    time, values = time[unique], values[unique]

    if args.normalize:
        center = np.nanmedian(values, axis=0)
        scale = 1.4826 * np.nanmedian(np.abs(values - center), axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.)
        values = (values - center) / scale

    figure, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True,
                                constrained_layout=True)
    colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00")
    for index, (axis, name, color) in enumerate(zip(axes, columns, colors)):
        axis.plot(time, values[:, index], color=color, linewidth=.8)
        axis.set_ylabel(name, rotation=0, ha="right", va="center")
        axis.grid(True, alpha=.22, linewidth=.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel(xlabel)
    figure.supylabel("Robust z-score" if args.normalize else "Recorded EMG")
    figure.suptitle(args.trial_csv.name)

    output = args.output or args.trial_csv.with_name(
        f"{args.trial_csv.stem}_four_emg.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    print(f"Saved {output}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
