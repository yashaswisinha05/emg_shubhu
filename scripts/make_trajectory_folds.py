#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from emg_touch.config import load_config
from emg_touch.data.manifest import load_manifest
from emg_touch.data.schema import natural_configuration_key
from emg_touch.utils import save_json


def stable_subject_seed(base_seed: int, configuration: str, subject: str) -> int:
    digest = hashlib.sha256(f"{configuration}:{subject}".encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def make_folds_for_configuration(
    configuration_frame,
    configuration: str,
    fold_count: int,
    seed: int,
    stratification_grid: tuple[int, int],
) -> list[dict]:
    subject_chunks: dict[str, list[list[str]]] = {}
    for subject, subject_frame in configuration_frame.groupby("subject", sort=True):
        if len(subject_frame) < fold_count:
            raise ValueError(
                f"{configuration}/{subject} has {len(subject_frame)} trials, "
                f"fewer than {fold_count} folds"
            )
        rng = np.random.default_rng(stable_subject_seed(seed, configuration, str(subject)))
        # Use coarse intended-target regions rather than noisy click coordinates for
        # strata. Targets are continuous, so exact-coordinate grouping would not balance
        # the screen. Repeated regions are distributed round-robin.
        grid_x, grid_y = stratification_grid
        target_x = subject_frame["target_x_norm"].to_numpy(dtype=np.float64)
        target_y = subject_frame["target_y_norm"].to_numpy(dtype=np.float64)
        if not np.isfinite(target_x).all() or not np.isfinite(target_y).all():
            raise ValueError(f"Non-finite target coordinates in {configuration}/{subject}")
        strata_frame = subject_frame.assign(
            _target_x=np.clip((target_x * grid_x).astype(int), 0, grid_x - 1),
            _target_y=np.clip((target_y * grid_y).astype(int), 0, grid_y - 1),
        )
        chunks: list[list[str]] = [[] for _ in range(fold_count)]
        for _, target_frame in strata_frame.groupby(
            ["_target_x", "_target_y"], sort=True, dropna=False
        ):
            trial_ids = target_frame["trial_id"].astype(str).to_numpy()
            shuffled = trial_ids[rng.permutation(len(trial_ids))]
            offset = int(rng.integers(0, fold_count))
            for index, trial_id in enumerate(shuffled):
                chunks[(offset + index) % fold_count].append(str(trial_id))
        if any(not chunk for chunk in chunks):
            raise ValueError(
                f"Target-stratified folding produced an empty fold for "
                f"{configuration}/{subject}"
            )
        subject_chunks[str(subject)] = chunks

    folds = []
    for fold_index in range(fold_count):
        validation_index = (fold_index + 1) % fold_count
        train, val, test = [], [], []
        counts = {}
        for subject, chunks in subject_chunks.items():
            subject_test = chunks[fold_index]
            subject_val = chunks[validation_index]
            subject_train = [
                trial_id
                for index, chunk in enumerate(chunks)
                if index not in {fold_index, validation_index}
                for trial_id in chunk
            ]
            train.extend(subject_train)
            val.extend(subject_val)
            test.extend(subject_test)
            counts[subject] = {
                "train": len(subject_train),
                "val": len(subject_val),
                "test": len(subject_test),
            }
        folds.append(
            {
                "configuration": configuration,
                "fold": fold_index,
                "split_strategy": "within_participant_target_stratified_5fold",
                "validation_fold": validation_index,
                "subject_counts": counts,
                "train": train,
                "val": val,
                "test": test,
            }
        )
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create full-trajectory cross-validation folds within every participant"
    )
    parser.add_argument("--config", default="configs/full_trajectory.yaml")
    parser.add_argument(
        "--configuration",
        action="append",
        help="Configuration to process; repeat as needed. Omit to process all configurations.",
    )
    parser.add_argument("--output-root", default="artifacts/trajectory_cv")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_manifest(config["paths"]["manifest"])
    available = sorted(manifest["configuration"].unique(), key=natural_configuration_key)
    selected = args.configuration or available
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown configurations: {sorted(unknown)}; available={available}")
    fold_count = int(config["cross_validation"]["folds"])
    grid = config["cross_validation"].get("stratification_grid", [8, 5])
    stratification_grid = (int(grid[0]), int(grid[1]))
    if stratification_grid[0] < 1 or stratification_grid[1] < 1:
        raise ValueError("cross_validation.stratification_grid must contain positive values")

    for configuration in selected:
        frame = manifest.loc[manifest["configuration"] == configuration]
        folds = make_folds_for_configuration(
            frame,
            configuration,
            fold_count,
            int(config["seed"]),
            stratification_grid,
        )
        for split in folds:
            output = (
                Path(args.output_root)
                / configuration
                / f"fold-{split['fold']}"
                / "split.json"
            )
            save_json(split, output)
            print(
                f"{output}: train={len(split['train'])}, "
                f"val={len(split['val'])}, test={len(split['test'])}"
            )


if __name__ == "__main__":
    main()
