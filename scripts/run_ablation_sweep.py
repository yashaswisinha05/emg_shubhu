#!/usr/bin/env python3
"""Run the full model x ablation x seed matrix and summarise it in one table.

Two single runs already disagree with the assumption this project started
from: with every input the model reaches 2.90 cm, and with EMG and IMU
zeroed it reaches 2.35 cm. Removing the muscle signal made it better. That
is a strong claim, and a single seed is not enough to make it - the earlier
screen-coordinate work in this repository showed differences of exactly this
size coming and going with the seed.

So this runs the matrix rather than one cell of it:

    model    trajectory | anticipatory
    inputs   all | no EMG | no IMU | kinematics only
    seeds    however many are asked for

and reports mean +- spread across seeds, so a difference is only called real
when it is larger than the seed noise underneath it.

Runs are skipped if their results.json already exists, so an interrupted
sweep resumes instead of starting over.

    python scripts/run_ablation_sweep.py \\
        --root "/media/.../emg_imu_vive" \\
        --config configs/tracked_trajectory.yaml \\
        --cache-dir artifacts/tracked_cache \\
        --output-dir runs/sweep --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
import sys
from pathlib import Path

ABLATIONS = {
    "all-inputs": "",
    "no-emg": "emg",
    "no-imu": "imu",
    "kinematics-only": "emg,imu",
}


def run_one(
    script: Path,
    root: str,
    config: str,
    cache_dir: str,
    output: Path,
    model: str,
    ablate: str,
    seed: int,
    epochs: int,
    device: str,
) -> dict | None:
    results = output / "results.json"
    if results.exists():
        print(f"  [skip, already done] {output.name}")
        return json.loads(results.read_text()).get("test")

    command = [
        sys.executable, str(script),
        "--root", root, "--config", config, "--cache-dir", cache_dir,
        "--output-dir", str(output), "--device", device,
        "--epochs", str(epochs), "--model", model, "--seed", str(seed),
    ]
    if ablate:
        command += ["--ablate", ablate]
    print(f"  running {output.name} ...", flush=True)
    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0:
        print(f"  FAILED ({finished.returncode}):")
        print("   " + "\n   ".join(finished.stderr.strip().splitlines()[-6:]))
        return None
    if not results.exists():
        print("  finished but wrote no results.json")
        return None
    return json.loads(results.read_text()).get("test")


def summarise(values: list[float]) -> str:
    clean = [v for v in values if v is not None]
    if not clean:
        return "        n/a"
    if len(clean) == 1:
        return f"{clean[0] * 100:7.2f}    "
    return f"{statistics.mean(clean) * 100:7.2f}±{statistics.stdev(clean) * 100:4.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/tracked_trajectory.yaml")
    parser.add_argument("--cache-dir", default="artifacts/tracked_cache")
    parser.add_argument("--output-dir", default="runs/sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--models", nargs="+", default=["trajectory", "anticipatory"],
        choices=("trajectory", "anticipatory"),
    )
    parser.add_argument(
        "--ablations", nargs="+", default=list(ABLATIONS),
        choices=list(ABLATIONS),
    )
    args = parser.parse_args()

    script = Path(__file__).resolve().parent / "train_trajectory_model.py"
    root_out = Path(args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    combinations = list(itertools.product(args.models, args.ablations, args.seeds))
    print(f"{len(combinations)} run(s): {len(args.models)} model(s) x "
          f"{len(args.ablations)} ablation(s) x {len(args.seeds)} seed(s)\n")

    collected: dict[tuple[str, str], list[dict]] = {}
    for model, ablation, seed in combinations:
        output = root_out / f"{model}__{ablation}__seed{seed}"
        scores = run_one(
            script, args.root, args.config, args.cache_dir, output,
            model, ABLATIONS[ablation], seed, args.epochs, args.device,
        )
        if scores:
            collected.setdefault((model, ablation), []).append(scores)

    if not collected:
        print("\nno runs produced results")
        sys.exit(1)

    # The baselines are properties of the data, not of any model, so they are
    # identical across every cell - printed once rather than in every row.
    any_scores = next(iter(collected.values()))[0]
    print("\n" + "=" * 78)
    print("baselines (same data for every run)")
    print(f"  hold   {any_scores.get('hold_mean_m', 0) * 100:6.2f} cm mean | "
          f"{any_scores.get('hold_final_m', 0) * 100:6.2f} cm final")
    print(f"  linear {any_scores.get('linear_mean_m', 0) * 100:6.2f} cm mean | "
          f"{any_scores.get('linear_final_m', 0) * 100:6.2f} cm final")
    linear_mean = any_scores.get("linear_mean_m")

    print("\n" + "=" * 78)
    header = (
        f"{'model':13} {'inputs':17} {'mean cm':>13} {'final cm':>13} "
        f"{'better than':>12}"
    )
    print(header)
    print(f"{'':13} {'':17} {'':>13} {'':>13} {'linear':>12}")
    print("-" * len(header))
    print("  (positive = beats constant-velocity extrapolation; negative = worse)")
    rows = []
    for (model, ablation), runs in sorted(collected.items()):
        means = [r.get("model_mean_m") for r in runs]
        finals = [r.get("model_final_m") for r in runs]
        clean = [v for v in means if v is not None]
        average = statistics.mean(clean) if clean else None
        improvement = (
            f"{(linear_mean - average) / linear_mean * 100:+7.1f}%"
            if average and linear_mean else "      n/a"
        )
        rows.append((average if average else 1e9, model, ablation, means, finals, improvement))
    for _, model, ablation, means, finals, improvement in sorted(rows):
        print(f"{model:13} {ablation:17} {summarise(means):>13} "
              f"{summarise(finals):>13} {improvement:>12}")

    # EMG's contribution, read two independent ways. They should agree; if
    # they do not, the disentangled split is mismeasuring rather than
    # revealing something.
    print("\n" + "=" * 78)
    print("what EMG contributes, measured two ways")
    for model in args.models:
        full = collected.get((model, "all-inputs"))
        kinematic = collected.get((model, "kinematics-only"))
        if full and kinematic:
            a = statistics.mean([r["model_mean_m"] for r in full])
            b = statistics.mean([r["model_mean_m"] for r in kinematic])
            verdict = "HELPS" if b > a else "HURTS"
            print(f"  [{model}] ablation: all-inputs {a * 100:.2f} cm vs "
                  f"kinematics-only {b * 100:.2f} cm -> EMG+IMU {verdict} by "
                  f"{abs(a - b) * 100:.2f} cm")
    antic = collected.get(("anticipatory", "all-inputs"))
    if antic and antic[0].get("kinematic_only_latent_mean_m"):
        full = statistics.mean([r["model_mean_m"] for r in antic])
        muted = statistics.mean([r["kinematic_only_latent_mean_m"] for r in antic])
        print(f"  [latent] silencing z_ant costs {(muted - full) * 100:.2f} cm "
              f"-> EMG's unique share of the prediction")

    save = root_out / "sweep_summary.json"
    save.write_text(json.dumps(
        {f"{m}__{a}": runs for (m, a), runs in collected.items()}, indent=2
    ))
    print(f"\nwrote {save}")


if __name__ == "__main__":
    main()
