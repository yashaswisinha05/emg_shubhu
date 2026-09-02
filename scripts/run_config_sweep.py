#!/usr/bin/env python3
"""Leave-one-condition-out: train on every condition but one, test on it.

The random session split reports a single number averaged over whichever
three sessions happened to land in the test set. That hides the thing worth
knowing about a dataset recorded under fourteen conditions: whether the model
transfers to a posture or task variant it has never seen, and whether some
conditions are simply harder than others.

Each condition here is one session, so holding a condition out is a
leave-one-session-out split labelled by what actually differs between the
sessions rather than by a hash.

    python scripts/run_config_sweep.py \\
        --root "/media/.../emg_imu_vive" \\
        --config configs/tracked_trajectory_emg_enhanced.yaml \\
        --cache-dir artifacts/tracked_cache \\
        --output-dir runs/config_sweep

Reports, per condition and per input set, the screen error in pixels against
its own `mean` and `centre` baselines. Those baselines are computed from the
held-out condition itself, so a condition whose targets happen to cluster is
not credited for being easy - the comparison is always against what guessing
would score on that same condition.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

CONFIG_PATTERN = re.compile(r"(?:^|[_-])(mix\d+|[ab]\d+)(?:_|$)", re.IGNORECASE)


def discover_configurations(root: Path) -> dict[str, int]:
    found: dict[str, int] = {}
    for path in root.rglob("trial_*.csv"):
        label = "unknown"
        for part in reversed(path.parts):
            match = CONFIG_PATTERN.search(part)
            if match:
                label = match.group(1).lower()
                break
        found[label] = found.get(label, 0) + 1
    return dict(sorted(found.items()))


def run_one(script, root, config, cache_dir, output, inputs, holdout, epochs,
            device, seed) -> dict | None:
    results = output / "results.json"
    if results.exists():
        return json.loads(results.read_text()).get("test")
    command = [
        sys.executable, str(script), "--root", root, "--config", config,
        "--cache-dir", cache_dir, "--output-dir", str(output),
        "--device", device, "--epochs", str(epochs), "--inputs", inputs,
        "--holdout-config", holdout,
    ]
    if seed is not None:
        command += ["--seed", str(seed)]
    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0:
        tail = finished.stderr.strip().splitlines()[-4:]
        print(f"    FAILED: {' | '.join(tail)}")
        return None
    if not results.exists():
        return None
    return json.loads(results.read_text()).get("test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/tracked_trajectory.yaml")
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache")
    parser.add_argument("--output-dir", default="runs/config_sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--inputs", nargs="+", default=["emg", "emg+imu"],
                        choices=("emg", "emg+imu"))
    parser.add_argument("--configs", nargs="+",
                        help="Conditions to hold out. Default: all discovered.")
    parser.add_argument(
        "--script", default="train_reach_target_model.py",
        help="Training script to drive - must accept --root --config "
        "--cache-dir --output-dir --device --epochs --inputs "
        "--holdout-config --seed, matching train_reach_target_model.py's "
        "CLI. train_grid_reach_model.py is a drop-in replacement.",
    )
    args = parser.parse_args()

    script = Path(__file__).resolve().parent / args.script
    root_out = Path(args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    discovered = discover_configurations(Path(args.root))
    print("conditions found: " + ", ".join(
        f"{name}({count})" for name, count in discovered.items()
    ))
    # 'unknown' is skipped by default: it is not one condition but every
    # session whose folder did not carry a recognisable label, so holding it
    # out would test transfer to a mixture rather than to a condition.
    chosen = args.configs or [n for n in discovered if n != "unknown"]
    print(f"holding out {len(chosen)} condition(s), {len(args.inputs)} input set(s)\n")

    collected: dict[tuple[str, str], dict] = {}
    for inputs in args.inputs:
        for label in chosen:
            output = root_out / f"{inputs.replace('+', '_')}__{label}"
            print(f"  {inputs:8} holdout={label} ...", flush=True)
            scores = run_one(
                script, args.root, args.config, args.cache_dir, output,
                inputs, label, args.epochs, args.device, args.seed,
            )
            if scores:
                collected[(inputs, label)] = scores

    if not collected:
        print("\nno runs produced results")
        sys.exit(1)

    for inputs in args.inputs:
        rows = {k[1]: v for k, v in collected.items() if k[0] == inputs}
        if not rows:
            continue
        # Two scripts can produce this summary, with different key names:
        # train_reach_target_model.py (model_screen_px/mean_screen_px, plus
        # a 3-D endpoint) and train_grid_reach_model.py (direct_px/mean_px,
        # screen-only, no endpoint). Both are read here rather than assuming
        # one script's schema, so --script is a genuine drop-in rather than
        # something that silently prints an empty table for the other.
        any_score = next(iter(rows.values()))
        screen_key = "model_screen_px" if "model_screen_px" in any_score else "direct_px"
        mean_key = "mean_screen_px" if "mean_screen_px" in any_score else "mean_px"
        centre_key = "centre_screen_px" if "centre_screen_px" in any_score else "centre_px"
        has_endpoint = "model_endpoint_m" in any_score

        print("\n" + "=" * 82)
        print(f"inputs: {inputs}   (tracker excluded in every run)")
        header = f"  {'condition':11}{'screen px':>11}{'vs mean':>10}{'vs centre':>11}"
        if has_endpoint:
            header += f"{'endpoint':>12}{'vs mean':>10}"
        print(header)
        print("  " + "-" * (63 if has_endpoint else 32))
        gains, endpoint_gains = [], []
        for label in sorted(rows):
            score = rows[label]
            model = score.get(screen_key)
            mean = score.get(mean_key)
            centre = score.get(centre_key)
            if model is None or mean is None:
                continue
            gain = (mean - model) / mean * 100
            centre_gain = (centre - model) / centre * 100 if centre else float("nan")
            gains.append(gain)
            row = f"  {label:11}{model:>11.1f}{gain:>+9.1f}%{centre_gain:>+10.1f}%"
            if has_endpoint:
                end_model = score.get("model_endpoint_m")
                end_mean = score.get("mean_endpoint_m")
                end_gain = (
                    (end_mean - end_model) / end_mean * 100
                    if end_model is not None and end_mean else float("nan")
                )
                endpoint_gains.append(end_gain)
                row += f"{end_model * 100:>10.2f} cm{end_gain:>+9.1f}%"
            print(row)
        if gains:
            print("  " + "-" * (63 if has_endpoint else 32))
            summary = f"  {'mean':11}{'':>11}{statistics.mean(gains):>+9.1f}%"
            if has_endpoint and endpoint_gains:
                summary += f"{'':>11}{'':>12}{statistics.mean(endpoint_gains):>+9.1f}%"
            print(summary)
            if len(gains) > 1:
                spread = f"  {'spread':11}{'':>11}{statistics.stdev(gains):>10.1f}"
                if has_endpoint and len(endpoint_gains) > 1:
                    spread += f"{'':>11}{'':>12}{statistics.stdev(endpoint_gains):>10.1f}"
                print(spread)

    print("\n" + "=" * 82)
    print("Positive means better than guessing ON THAT CONDITION. The baselines")
    print("are recomputed per condition, so a condition with tightly clustered")
    print("targets is not credited for being easy - it is scored against what")
    print("guessing would achieve there.")

    save = root_out / "config_sweep_summary.json"
    save.write_text(json.dumps(
        {f"{i}__{c}": s for (i, c), s in collected.items()}, indent=2
    ))
    print(f"\nwrote {save}")


if __name__ == "__main__":
    main()
