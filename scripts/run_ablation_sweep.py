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

import yaml

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
    task: str = "forecast",
    extra_args: tuple[str, ...] = (),
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
        "--task", task,
    ]
    command += list(extra_args)
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
    parser.add_argument("--cutoff-offset-ms", type=float, nargs="+")
    parser.add_argument("--separate-modalities", action="store_true")
    parser.add_argument("--emg-only-weight", type=float)
    parser.add_argument("--imu-dropout", type=float)
    parser.add_argument("--emg-feature-windows-ms", type=float, nargs="+")
    parser.add_argument(
        "--emg-feature-kinds", nargs="+",
        choices=("rms", "waveform_length", "log_energy", "derivative"),
    )
    parser.add_argument("--paired-modality-interventions", action="store_true")
    parser.add_argument(
        "--task", choices=("forecast", "wearable"), default="forecast",
        help="wearable: tracker is label-only, EMG+IMU are the whole input. "
        "This is where the EMG-vs-IMU decomposition actually answers the "
        "project's question, since in forecast mode the tracker dominates "
        "both modalities.",
    )
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

    if args.task == "wearable":
        print("wearable: the tracker is already excluded, so the ablations "
              "compare EMG against IMU rather than against the tracker\n")
    combinations = list(itertools.product(args.models, args.ablations, args.seeds))
    print(f"{len(combinations)} run(s): {len(args.models)} model(s) x "
          f"{len(args.ablations)} ablation(s) x {len(args.seeds)} seed(s)\n")

    collected: dict[tuple[str, str], list[dict]] = {}
    extra_args: list[str] = []
    if args.cutoff_offset_ms:
        extra_args += ["--cutoff-offset-ms", *map(str, args.cutoff_offset_ms)]
    if args.separate_modalities:
        extra_args.append("--separate-modalities")
    if args.emg_only_weight is not None:
        extra_args += ["--emg-only-weight", str(args.emg_only_weight)]
    if args.imu_dropout is not None:
        extra_args += ["--imu-dropout", str(args.imu_dropout)]
    if args.emg_feature_windows_ms:
        extra_args += ["--emg-feature-windows-ms", *map(str, args.emg_feature_windows_ms)]
    if args.emg_feature_kinds:
        extra_args += ["--emg-feature-kinds", *args.emg_feature_kinds]
    if args.paired_modality_interventions:
        extra_args.append("--paired-modality-interventions")
    for model, ablation, seed in combinations:
        output = root_out / f"{model}__{ablation}__seed{seed}"
        scores = run_one(
            script, args.root, args.config, args.cache_dir, output,
            model, ABLATIONS[ablation], seed, args.epochs, args.device, args.task,
            tuple(extra_args),
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
    if "mean_reach_mean_m" in any_scores:
        print(f"  mean   {any_scores['mean_reach_mean_m'] * 100:6.2f} cm mean | "
              f"{any_scores.get('mean_reach_final_m', 0) * 100:6.2f} cm final"
              "   <- the bar in wearable mode")
    # In wearable mode `linear` needs the tracker, so it cannot be the
    # reference; the average reach profile is what a model must beat to have
    # inferred anything about THIS trial.
    linear_mean = any_scores.get("mean_reach_mean_m") or any_scores.get("linear_mean_m")

    print("\n" + "=" * 78)
    reference_name = (
        "mean reach" if "mean_reach_mean_m" in any_scores else "linear"
    )
    header = (
        f"{'model':13} {'inputs':17} {'mean cm':>13} {'final cm':>13} "
        f"{'better than':>12}"
    )
    print(header)
    print(f"{'':13} {'':17} {'':>13} {'':>13} {reference_name:>12}")
    print("-" * len(header))
    if reference_name == "mean reach":
        print("  (positive = infers something about THIS trial rather than "
              "replaying the average reach)")
    else:
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

    # These are interventions on the SAME trained checkpoint, unlike the
    # separately trained ablation rows above. Positive costs mean destroying
    # that modality made prediction worse; the shuffled-EMG cost is the
    # cleanest check that the model uses trial-specific EMG rather than its
    # marginal amplitude distribution.
    if any("without_emg_mean_m" in run for runs in collected.values() for run in runs):
        print("\n" + "=" * 78)
        print("paired same-checkpoint modality effects (positive = modality helps)")
        print(f"{'model':13} {'inputs':17} {'remove EMG':>13} "
              f"{'shuffle EMG':>13} {'remove IMU':>13}")
        print("-" * 78)
        for (model, ablation), runs in sorted(collected.items()):
            def costs(intervention: str) -> list[float]:
                return [
                    run[f"{intervention}_mean_m"] - run["model_mean_m"]
                    for run in runs
                    if run.get(f"{intervention}_mean_m") is not None
                    and run.get("model_mean_m") is not None
                ]

            print(
                f"{model:13} {ablation:17} {summarise(costs('without_emg')):>13} "
                f"{summarise(costs('shuffled_emg')):>13} "
                f"{summarise(costs('without_imu')):>13}"
            )

    if args.cutoff_offset_ms:
        print("\n" + "=" * 78)
        print("intact model by exact cutoff relative to movement onset")
        print(f"{'model':13} {'inputs':17}" + "".join(
            f"{value:+.0f} ms".rjust(13) for value in args.cutoff_offset_ms
        ))
        print("-" * 78)
        raw_config = yaml.safe_load(Path(args.config).read_text())
        rate = (
            float(raw_config["data"]["sample_rate_hz"])
            / int(raw_config["data"]["decimation"])
        )
        sample_keys = [
            f"{int(round(value * rate / 1000.0)):+d}"
            for value in args.cutoff_offset_ms
        ]
        for (model, ablation), runs in sorted(collected.items()):
            cells = []
            for key in sample_keys:
                values = [run.get(f"model_offset_{key}_m") for run in runs]
                cells.append(summarise([value for value in values if value is not None]))
            print(f"{model:13} {ablation:17}" + "".join(f"{cell:>13}" for cell in cells))

    # Errors split by how early the cutoff sat. Electromechanical delay puts
    # any EMG contribution in the first samples after onset, so an effect can
    # be real there and invisible in a mean over uniformly sampled cutoffs.
    buckets = ("early", "mid", "late")
    if any(f"model_{b}_m" in any_scores for b in buckets):
        print("\n" + "=" * 78)
        print("by how early the cutoff sat (samples past movement onset)")
        print(f"{'model':13} {'inputs':17}" + "".join(f"{b:>12}" for b in buckets))
        print("-" * 78)
        for (model, ablation), runs in sorted(collected.items()):
            cells = []
            for bucket in buckets:
                values = [r.get(f"model_{bucket}_m") for r in runs]
                values = [v for v in values if v is not None]
                cells.append(f"{statistics.mean(values) * 100:9.2f} cm" if values else "         -")
            print(f"{model:13} {ablation:17}" + "".join(f"{c:>12}" for c in cells))

    # Separating the two modalities. Ablating EMG and IMU together cannot say
    # which one is responsible - a 2x2 can, and measuring each main effect
    # twice (once with the other modality present, once without) shows whether
    # it is consistent or an artifact of one particular combination.
    print("\n" + "=" * 78)
    print("separating EMG from IMU (each effect measured twice)")

    def mean_of(model: str, ablation: str) -> float | None:
        runs = collected.get((model, ablation))
        if not runs:
            return None
        values = [r.get("model_mean_m") for r in runs]
        values = [v for v in values if v is not None]
        return statistics.mean(values) if values else None

    def spread_of(model: str, ablation: str) -> float:
        runs = collected.get((model, ablation)) or []
        values = [r.get("model_mean_m") for r in runs]
        values = [v for v in values if v is not None]
        return statistics.stdev(values) if len(values) > 1 else 0.0

    for model in args.models:
        both = mean_of(model, "all-inputs")
        without_imu = mean_of(model, "no-imu")
        without_emg = mean_of(model, "no-emg")
        neither = mean_of(model, "kinematics-only")
        noise = max(
            spread_of(model, a)
            for a in ("all-inputs", "no-emg", "no-imu", "kinematics-only")
        )
        if None in (both, without_imu, without_emg, neither):
            continue
        # Cost of keeping a modality: positive means the model is better off
        # without it.
        emg_with_imu = both - without_emg
        emg_without_imu = without_imu - neither
        imu_with_emg = both - without_imu
        imu_without_emg = without_emg - neither
        print(f"\n  [{model}]  seed noise ~{noise * 100:.2f} cm")
        for label, a, b in (
            ("EMG", emg_with_imu, emg_without_imu),
            ("IMU", imu_with_emg, imu_without_emg),
        ):
            average = (a + b) / 2
            sigma = abs(average) / noise if noise > 0 else float("inf")
            if sigma < 2.0:
                verdict = "within seed noise - no effect"
            elif average > 0:
                verdict = f"HURTS by {average * 100:.2f} cm ({sigma:.1f} sd)"
            else:
                verdict = f"HELPS by {-average * 100:.2f} cm ({sigma:.1f} sd)"
            print(f"    {label}: {a * 100:+.2f} / {b * 100:+.2f} cm  -> {verdict}")
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
