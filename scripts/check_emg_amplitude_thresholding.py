#!/usr/bin/env python3
"""Diagnostic only: can naive EMG-amplitude thresholding find grasp on/offset?

No training, no model, nothing written back to the dataset. For every
trial*.csv under --root this:

  1. reuses emg_touch.data.reach_grasp.preprocess (the same column parsing,
     event-time resolution, causal band-pass + trailing RMS, and gap-aware
     validity masking every training script in this repo already trusts) to
     get a per-channel amplitude envelope on a uniform time grid, aligned to
     grasp_onset_s / grasp_offset_s (or absolute t_grasp_perf/t_release_perf);
  2. smooths each channel with a rolling median (--median-window-s);
  3. builds two channel COMBINATIONS -- sum and max across the 4 sensors --
     since a single channel may not carry the pattern even if the ensemble
     does;
  4. for each of {channel 0..3, sum, max} and a grid of threshold multipliers
     k, sets threshold = baseline_mean + k * baseline_std, where the baseline
     comes from the first --baseline-window-s seconds of THAT trial (assumed
     to be the pre-reach rest posture -- see the caveat printed at the end if
     that assumption looks wrong for this dataset);
  5. detects the onset as the first sustained (>= --persistence-s) crossing
     above threshold inside a padded search window around the annotated
     grasp interval, and the offset as the first sustained drop below
     threshold * --low-ratio afterwards (hysteresis, matching the decoder
     convention already used in scripts/inspect_grasp_triggers.py);
  6. flags any OTHER sustained crossing in the trial outside the annotated
     interval (+/- the search pad) as a false trigger -- this is the number
     that matters most for real reach-then-grasp trials, where forearm EMG
     during the reach can already be well above a quiet-rest threshold.

Aggregated across all trials it reports, per (signal, k): hit rate, median
|onset error| and |offset error| in ms, and mean false triggers per trial --
then a recommended k per signal and a plain verdict on whether simple
amplitude thresholding is viable here at all, or whether it only looks viable
until you count false triggers during the reach.

    python scripts/check_emg_amplitude_thresholding.py \\
        --root /home/nahar3/shubham/emg_shubhu/data/184a6ef69b83

    # with example plots
    python scripts/check_emg_amplitude_thresholding.py \\
        --root /home/nahar3/shubham/emg_shubhu/data/184a6ef69b83 \\
        --plots 6 --output-dir runs/emg_amplitude_thresholding_184a6ef69b83
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emg_touch.data.reach_grasp import SENSORS, preprocess  # noqa: E402

SIGNAL_NAMES = [f"ch_{s}" for s in SENSORS] + ["sum", "max"]


def rolling_median(x: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median per column, NaN-holed at invalid samples first."""
    if window <= 1:
        return x
    holed = np.where(valid, x, np.nan)
    return (
        pd.DataFrame(holed)
        .rolling(window, center=True, min_periods=1)
        .median()
        .fillna(0.0)
        .to_numpy()
    )


def build_signals(short_rms: np.ndarray, valid: np.ndarray, median_window: int):
    """Per-channel median-filtered envelopes plus sum/max combinations."""
    smoothed = rolling_median(short_rms, valid, median_window)
    all_valid = valid.all(axis=1)
    signals = {f"ch_{s}": smoothed[:, i] for i, s in enumerate(SENSORS)}
    signals["sum"] = smoothed.sum(axis=1)
    signals["max"] = smoothed.max(axis=1)
    return signals, all_valid


def baseline_stats(signal: np.ndarray, valid: np.ndarray, time: np.ndarray,
                    window_s: float, pad_s: float) -> tuple[float, float] | None:
    mask = valid & (time >= time[0] + pad_s) & (time < time[0] + pad_s + window_s)
    if mask.sum() < 5:
        return None
    values = signal[mask]
    return float(values.mean()), float(max(values.std(), 1e-9))


def sustained_crossings(signal: np.ndarray, valid: np.ndarray, dt: float,
                         high: float, low: float, persistence_s: float):
    """All (rise_time_idx, fall_time_idx) pairs where signal >= high for
    >= persistence_s, ended by a sustained drop below low. Invalid samples
    break any pending evidence, matching this repo's stated decoding policy
    (docs/reach_grasp_annotation.md: "missing data interrupts pending
    evidence")."""
    need = max(1, int(round(persistence_s / dt)))
    above = valid & (signal >= high)
    below = valid & (signal < low)
    events = []
    i, n = 0, len(signal)
    while i < n:
        if above[i]:
            run_start = i
            j = i
            while j < n and above[j]:
                j += 1
            if j - run_start >= need:
                rise = run_start + need - 1
                k = j
                fall = None
                while k < n:
                    if not valid[k]:
                        k += 1
                        continue
                    if below[k]:
                        run2 = k
                        m = k
                        while m < n and below[m]:
                            m += 1
                        if m - run2 >= need:
                            fall = run2 + need - 1
                            k = m
                            break
                        k = m
                    else:
                        k += 1
                events.append((rise, fall if fall is not None else n - 1))
                i = k if fall is not None else n
                continue
            i = j
        else:
            i += 1
    return events


def evaluate_trial(trial: dict, median_window_samples: int, baseline_window_s: float,
                    baseline_pad_s: float, search_pad_s: float, persistence_s: float,
                    low_ratio: float, k_grid: np.ndarray, min_valid_fraction: float):
    time = trial["time"]
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
    onset_true, offset_true = trial["events"]
    short_rms = trial["emg"][:, :4]
    valid = trial["emg_valid"][:, :4]
    if valid.mean() < min_valid_fraction:
        return None, "too much missing data"

    signals, all_valid = build_signals(short_rms, valid, median_window_samples)

    search_lo = onset_true - search_pad_s
    search_hi = offset_true + search_pad_s
    in_search = (time >= search_lo) & (time <= search_hi)

    per_signal = {}
    for name in SIGNAL_NAMES:
        signal = signals[name]
        stats = baseline_stats(signal, all_valid, time, baseline_window_s, baseline_pad_s)
        if stats is None:
            per_signal[name] = None
            continue
        mean, std = stats
        per_signal[name] = {"mean": mean, "std": std, "signal": signal}

    per_k = {name: [] for name in SIGNAL_NAMES}
    for name in SIGNAL_NAMES:
        info = per_signal[name]
        if info is None:
            for k in k_grid:
                per_k[name].append({"k": float(k), "hit": False, "onset_error_ms": None,
                                     "offset_error_ms": None, "false_triggers": None})
            continue
        signal, mean, std = info["signal"], info["mean"], info["std"]
        for k in k_grid:
            high = mean + k * std
            low = mean + k * std * low_ratio
            events = sustained_crossings(signal, all_valid, dt, high, low, persistence_s)
            in_window = [(r, f) for r, f in events if in_search[r]]
            outside = [(r, f) for r, f in events if not in_search[r]]
            if not in_window:
                per_k[name].append({"k": float(k), "hit": False, "onset_error_ms": None,
                                     "offset_error_ms": None,
                                     "false_triggers": len(outside)})
                continue
            rise, fall = min(in_window, key=lambda e: abs(time[e[0]] - onset_true))
            onset_err = (time[rise] - onset_true) * 1000.0
            offset_err = (time[fall] - offset_true) * 1000.0
            per_k[name].append({"k": float(k), "hit": True,
                                 "onset_error_ms": float(onset_err),
                                 "offset_error_ms": float(offset_err),
                                 "false_triggers": len(outside) + max(0, len(in_window) - 1)})

    return {"path": trial["path"], "time": time, "events": (onset_true, offset_true),
            "signals": {n: (per_signal[n]["signal"] if per_signal[n] else None)
                        for n in SIGNAL_NAMES},
            "baselines": {n: (per_signal[n]["mean"], per_signal[n]["std"]) if per_signal[n] else None
                          for n in SIGNAL_NAMES},
            "baseline_available": {n: per_signal[n] is not None for n in SIGNAL_NAMES},
            "per_k": per_k}, None


def aggregate(results: list[dict], k_grid: np.ndarray) -> dict:
    agg = {}
    for name in SIGNAL_NAMES:
        agg.setdefault("_baseline_available_fraction", {})[name] = float(
            np.mean([r["baseline_available"][name] for r in results])
        ) if results else 0.0
        rows = []
        for k in k_grid:
            hits, onset_e, offset_e, false_trig = [], [], [], []
            for r in results:
                entry = next(e for e in r["per_k"][name] if abs(e["k"] - k) < 1e-9)
                hits.append(entry["hit"])
                if entry["hit"]:
                    onset_e.append(entry["onset_error_ms"])
                    offset_e.append(entry["offset_error_ms"])
                if entry["false_triggers"] is not None:
                    false_trig.append(entry["false_triggers"])
            hit_rate = float(np.mean(hits)) if hits else 0.0
            rows.append({
                "k": float(k),
                "hit_rate": hit_rate,
                "median_abs_onset_error_ms": float(np.median(np.abs(onset_e))) if onset_e else None,
                "median_abs_offset_error_ms": float(np.median(np.abs(offset_e))) if offset_e else None,
                "mean_false_triggers": float(np.mean(false_trig)) if false_trig else None,
            })
        agg[name] = rows
    return agg


def recommend(agg_rows: list[dict], min_hit_rate: float = 0.85) -> dict | None:
    candidates = [r for r in agg_rows if r["hit_rate"] >= min_hit_rate
                  and r["median_abs_onset_error_ms"] is not None]
    if not candidates:
        candidates = [r for r in agg_rows if r["hit_rate"] > 0]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r["hit_rate"])

    def score(r):
        timing = abs(r["median_abs_onset_error_ms"]) + abs(r["median_abs_offset_error_ms"] or 0.0)
        penalty = (r["mean_false_triggers"] or 0.0) * 50.0
        return timing + penalty

    return min(candidates, key=score)


def plot_trial(result: dict, k: float, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = result["time"]
    onset_true, offset_true = result["events"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for ch in SENSORS:
        axes[0].plot(time, result["signals"][f"ch_{ch}"], label=ch, alpha=0.8)
    axes[0].axvspan(onset_true, offset_true, color="tab:green", alpha=0.15, label="annotated hold")
    axes[0].set_ylabel("median-filtered RMS")
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].set_title(Path(result["path"]).name)

    for name, colour in (("sum", "tab:blue"), ("max", "tab:orange")):
        signal = result["signals"][name]
        mean, std = result["baselines"][name]
        axes[1].plot(time, signal, color=colour, label=name)
        axes[1].axhline(mean + k * std, color=colour, linestyle="--", linewidth=1,
                         label=f"{name} threshold (k={k:g})")
    axes[1].axvspan(onset_true, offset_true, color="tab:green", alpha=0.15)
    axes[1].set_ylabel("combined signal")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--pattern", default="**/trial*.csv",
                         help="glob relative to --root; use '**/*.csv' if trials are not prefixed 'trial'")
    parser.add_argument("--raw-rate-hz", type=float, default=1259.4)
    parser.add_argument("--rate-hz", type=float, default=200.0,
                         help="resampled diagnostic grid rate; finer than the 100 Hz used for training")
    parser.add_argument("--event-origin", choices=["auto", "start"], default="auto")
    parser.add_argument("--gap-s", type=float, default=0.02)
    parser.add_argument("--median-window-s", type=float, default=0.05)
    parser.add_argument("--baseline-window-s", type=float, default=0.3,
                         help="seconds from the START of each trial used as the rest baseline")
    parser.add_argument("--baseline-pad-s", type=float, default=0.0,
                         help="skip this many seconds at the very start before the baseline window")
    parser.add_argument("--search-pad-s", type=float, default=0.3,
                         help="search window around the annotated grasp interval; "
                              "anything outside it counts as a false trigger")
    parser.add_argument("--persistence-s", type=float, default=0.05)
    parser.add_argument("--low-ratio", type=float, default=0.6,
                         help="offset hysteresis: falls when signal < k*std*low_ratio")
    parser.add_argument("--k-min", type=float, default=1.5)
    parser.add_argument("--k-max", type=float, default=8.0)
    parser.add_argument("--k-step", type=float, default=0.5)
    parser.add_argument("--min-valid-fraction", type=float, default=0.75)
    parser.add_argument("--min-hit-rate", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--plots", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.root.glob(args.pattern))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no files matched {args.pattern!r} under {args.root}")

    settings = {"raw_rate_hz": args.raw_rate_hz, "rate_hz": args.rate_hz,
                "gap_s": args.gap_s, "event_origin": args.event_origin,
                "event_pulse_s": 0.1}
    median_window_samples = max(1, int(round(args.median_window_s * args.rate_hz)))
    k_grid = np.arange(args.k_min, args.k_max + 1e-9, args.k_step)

    results, skipped = [], []
    for path in paths:
        try:
            trial = preprocess(path, settings)
        except Exception as exc:  # noqa: BLE001 - report and continue past one bad trial
            skipped.append((path, f"preprocess failed: {exc}"))
            continue
        result, reason = evaluate_trial(
            trial, median_window_samples, args.baseline_window_s, args.baseline_pad_s,
            args.search_pad_s, args.persistence_s, args.low_ratio, k_grid,
            args.min_valid_fraction,
        )
        if result is None:
            skipped.append((path, reason))
            continue
        results.append(result)

    print(f"{len(results)} trials evaluated, {len(skipped)} skipped, out of {len(paths)} found")
    if skipped:
        by_reason: dict[str, int] = {}
        for _, reason in skipped:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            print(f"  skipped {count:3d}  {reason}")
    if not results:
        raise SystemExit("no trials could be evaluated -- check --pattern and --event-origin")

    holds = [r["events"][1] - r["events"][0] for r in results]
    print(f"\nannotated hold duration: median {np.median(holds):.2f}s "
          f"(min {min(holds):.2f}s, max {max(holds):.2f}s)\n")

    agg = aggregate(results, k_grid)

    print(f"{'signal':8s} {'best k':>7s} {'hit rate':>9s} "
          f"{'|onset err| ms':>15s} {'|offset err| ms':>16s} {'false trig/trial':>17s}")
    recommendations = {}
    for name in SIGNAL_NAMES:
        best = recommend(agg[name], args.min_hit_rate)
        recommendations[name] = best
        if best is None:
            available = agg["_baseline_available_fraction"][name]
            if available == 0.0:
                print(f"{name:8s}   -- no trial had a usable rest baseline for this signal --")
            else:
                print(f"{name:8s}   -- baseline OK on {available*100:.0f}% of trials, but no "
                      f"k in [{args.k_min:g}, {args.k_max:g}] ever matched the annotated window --")
            continue
        onset = best["median_abs_onset_error_ms"]
        offset = best["median_abs_offset_error_ms"]
        false_trig = best["mean_false_triggers"]
        print(f"{name:8s} {best['k']:7.1f} {best['hit_rate']*100:8.1f}% "
              f"{onset if onset is not None else float('nan'):15.1f} "
              f"{offset if offset is not None else float('nan'):16.1f} "
              f"{false_trig if false_trig is not None else float('nan'):17.2f}")

    ranked = sorted(
        ((n, r) for n, r in recommendations.items() if r is not None),
        key=lambda nr: (nr[1]["mean_false_triggers"] or 1e9, nr[1]["median_abs_onset_error_ms"] or 1e9),
    )
    print()
    if not ranked:
        if all(agg["_baseline_available_fraction"][n] == 0.0 for n in SIGNAL_NAMES):
            print("VERDICT: no signal ever got a usable rest baseline -- "
                  "check that --baseline-window-s actually lands on rest for this dataset "
                  "(try --baseline-pad-s or a --plots example first).")
        else:
            print(f"VERDICT: baselines are fine but no threshold in "
                  f"[{args.k_min:g}, {args.k_max:g}] ever matched the annotated window on "
                  "any trial -- try widening --k-max, or amplitude genuinely does not carry "
                  "this pattern and it is worth checking co-activation / envelope-ripple "
                  "features instead.")
    else:
        best_name, best = ranked[0]
        low_trigger = (best["mean_false_triggers"] or 0.0) <= 0.3
        good_hit = best["hit_rate"] >= args.min_hit_rate
        good_timing = (best["median_abs_onset_error_ms"] or 1e9) <= 150.0
        if good_hit and good_timing and low_trigger:
            print(f"VERDICT: simple thresholding on '{best_name}' (k={best['k']:g}) looks "
                  f"viable -- {best['hit_rate']*100:.0f}% hit rate, "
                  f"{best['median_abs_onset_error_ms']:.0f} ms median onset error, "
                  f"{best['mean_false_triggers']:.2f} false triggers/trial.")
        elif good_hit and good_timing:
            print(f"VERDICT: '{best_name}' finds onset/offset well "
                  f"({best['median_abs_onset_error_ms']:.0f} ms) but still fires "
                  f"{best['mean_false_triggers']:.2f} times/trial outside the annotated window "
                  "-- almost certainly the reach phase, not the grasp. A threshold this simple "
                  "is not sufficient on its own; it needs a state machine or a hold-duration "
                  "gate on top.")
        else:
            print(f"VERDICT: no signal/threshold combination reaches "
                  f"{args.min_hit_rate*100:.0f}% hit rate with clean timing. Best found: "
                  f"'{best_name}' at {best['hit_rate']*100:.0f}% hit rate. Amplitude alone "
                  "does not appear to carry a clean onset pattern in this dataset -- "
                  "check the co-activation / envelope-ripple features instead.")

    print("\nCAVEAT: the baseline is the first "
          f"{args.baseline_window_s:g}s of each trial, assumed to be rest before the reach "
          "begins. If trials are cropped to start mid-reach, every number above is measuring "
          "the wrong thing -- rerun with --baseline-pad-s / --baseline-window-s adjusted, or "
          "inspect a --plots example to check.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "root": str(args.root), "pattern": args.pattern, "n_trials": len(results),
            "n_skipped": len(skipped), "k_grid": k_grid.tolist(), "aggregate": agg,
            "recommendations": {n: r for n, r in recommendations.items()},
            "settings": vars(args) | {"root": str(args.root), "output_dir": str(args.output_dir)},
        }
        with (args.output_dir / "report.json").open("w") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nwrote {args.output_dir / 'report.json'}")

    if args.plots:
        plot_dir = args.output_dir or Path("emg_amplitude_thresholding_plots")
        plot_k = ranked[0][1]["k"] if ranked else float(k_grid[len(k_grid) // 2])
        for result in results[: args.plots]:
            out_path = plot_dir / f"{Path(result['path']).stem}.png"
            plot_trial(result, plot_k, out_path)
        print(f"wrote {min(args.plots, len(results))} plots -> {plot_dir}")


if __name__ == "__main__":
    main()
