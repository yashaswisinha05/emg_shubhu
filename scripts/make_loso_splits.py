#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from emg_touch.config import load_config
from emg_touch.data.manifest import load_manifest
from emg_touch.data.splits import make_subject_split
from emg_touch.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create rotating leave-one-subject-out folds")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--output-root", default="artifacts/loso")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Used only when a configuration has exactly two participants",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    frame = load_manifest(config["paths"]["manifest"])
    selected = frame.loc[frame["configuration"] == args.configuration]
    subjects = sorted(selected["subject"].unique())
    if len(subjects) < 2:
        raise ValueError("At least two participants are required for held-out evaluation")
    for index, test_subject in enumerate(subjects):
        if len(subjects) >= 3:
            val_subject = subjects[(index + 1) % len(subjects)]
            split = make_subject_split(selected, test_subject, val_subject, args.configuration)
            validation_mode = "held_out_participant"
        else:
            training_pool = selected.loc[selected["subject"] != test_subject].copy()
            test = selected.loc[selected["subject"] == test_subject, "trial_id"].tolist()
            rng = np.random.default_rng(int(config["seed"]) + index)
            order = rng.permutation(len(training_pool))
            val_count = max(1, int(round(len(training_pool) * args.validation_fraction)))
            val_indices = order[:val_count]
            train_indices = order[val_count:]
            if len(train_indices) == 0:
                raise ValueError("Validation fraction leaves no training trials")
            split = {
                "configuration": args.configuration,
                "test_subject": test_subject,
                "val_subject": None,
                "validation_mode": "training_participant_trial_split",
                "train": training_pool.iloc[train_indices]["trial_id"].tolist(),
                "val": training_pool.iloc[val_indices]["trial_id"].tolist(),
                "test": test,
            }
            val_subject = "training-participant trials"
            validation_mode = split["validation_mode"]
        output = Path(args.output_root) / args.configuration / f"test-{test_subject}" / "split.json"
        save_json(split, output)
        print(
            f"{output}: test={test_subject}, val={val_subject}, "
            f"validation_mode={validation_mode}, train_trials={len(split['train'])}"
        )


if __name__ == "__main__":
    main()
