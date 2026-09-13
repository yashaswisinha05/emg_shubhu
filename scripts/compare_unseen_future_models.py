#!/usr/bin/env python3
"""Evaluate proposed, GRU, and LSTM checkpoints on identical unseen roots."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def model_spec(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("use NAME=CHECKPOINT")
    name, checkpoint = value.split("=", 1)
    if not name or not checkpoint:
        raise argparse.ArgumentTypeError("use NAME=CHECKPOINT")
    return name, Path(checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=model_spec, required=True,
                        metavar="NAME=CHECKPOINT")
    parser.add_argument("--root", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--velocity-lookback-ms", type=float, default=200.)
    args = parser.parse_args()
    names = [name for name, _ in args.model]
    if len(names) != len(set(names)):
        parser.error("model names must be unique")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reports = {}
    evaluator = Path(__file__).with_name("evaluate_residual_gru_roots.py")
    for name, checkpoint in args.model:
        if not checkpoint.is_file():
            parser.error(f"checkpoint not found for {name}: {checkpoint}")
        temporary = args.output.parent / f".{args.output.stem}_{name}.json"
        command = [sys.executable, str(evaluator), "--checkpoint", str(checkpoint),
                   "--root", *args.root, "--device", args.device,
                   "--velocity-lookback-ms", str(args.velocity_lookback_ms),
                   "--output", str(temporary)]
        subprocess.run(command, check=True)
        reports[name] = json.loads(temporary.read_text())
        temporary.unlink()

    accepted = {name: report["trials"]["accepted"] for name, report in reports.items()}
    if len(set(accepted.values())) != 1:
        print(f"WARNING: accepted trial counts differ: {accepted}", file=sys.stderr)
    result = {
        "protocol": {
            "roots": args.root,
            "velocity_baseline": "past-only least-squares constant velocity",
            "velocity_lookback_ms": args.velocity_lookback_ms,
            "note": ("Only horizons emitted by each checkpoint are compared; "
                     "the proposed model may additionally emit 1-s intent."),
        },
        "models": reports,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Comparison written to {args.output}")


if __name__ == "__main__":
    main()
