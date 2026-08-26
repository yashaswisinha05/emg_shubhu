#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emg_touch.checkpointing import load_model_state
from emg_touch.config import load_config
from emg_touch.data.loaders import build_loaders
from emg_touch.models.factory import build_model
from emg_touch.training import evaluate_model
from emg_touch.utils import choose_device, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint at causal cue-relative cutoffs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kind", choices=["teacher", "student", "tcn", "patchtst"], required=True)
    parser.add_argument("--split")
    parser.add_argument("--scaler")
    parser.add_argument("--cutoffs", type=float, nargs="*")
    parser.add_argument("--output-dir", default="evaluation")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = choose_device(args.device)
    model = build_model(args.kind, config).to(device)
    load_model_state(model, args.checkpoint)
    cutoffs = args.cutoffs if args.cutoffs is not None else config["data"]["eval_cutoffs_s"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records, summary = [], {}

    for cutoff in cutoffs:
        _, _, test_loader = build_loaders(
            config, args.split, args.scaler, eval_cutoff_s=float(cutoff)
        )
        metrics, records = evaluate_model(model, test_loader, args.kind, device)
        requested = "full" if float(cutoff) < 0 else f"{float(cutoff):.3f}"
        for record in records:
            record["requested_cutoff"] = requested
            record["model_kind"] = args.kind
        all_records.extend(records)
        summary[requested] = metrics
        print(requested, metrics)

    pd.DataFrame(all_records).to_csv(output_dir / "predictions.csv", index=False)
    save_json(summary, output_dir / "metrics.json")


if __name__ == "__main__":
    main()

