#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from emg_touch.config import load_config
from emg_touch.data.grid_trajectory import (
    extend_emg_mask,
    preprocess_grid_signals,
    raw_emg_features,
)
from emg_touch.data.manifest import load_manifest
from emg_touch.data.preprocessing import robust_statistics
from emg_touch.data.splits import subset_from_trial_ids
from emg_touch.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit training-only scaling for EMG and configured IMU features"
    )
    parser.add_argument("--config", default="configs/grid_point.yaml")
    parser.add_argument("--split", help="Override split JSON")
    parser.add_argument("--output", help="Override scaler NPZ")
    parser.add_argument("--samples-per-trial", type=int, default=128)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = load_manifest(config["paths"]["manifest"])
    split = load_json(args.split or config["paths"]["split_file"])
    train = subset_from_trial_ids(manifest, split["train"])
    rng = np.random.default_rng(int(config["seed"]))
    minimum = float(config["data"].get("min_duration_s", 0.0))
    maximum = float(config["data"].get("max_duration_s", float("inf")))
    imu_values: list[np.ndarray] = []
    # Per-trial untransformed EMG amplitudes plus their session id. A single
    # pooled scale cannot remove a per-session multiplicative gain, and EMG
    # amplitude varies 2.1x-7.3x between a1 sessions (electrode impedance,
    # skin preparation, adiposity, placement). Collected from TRAINING trials
    # only - using any validation or test trial here would leak.
    raw_emg: list[np.ndarray] = []
    raw_emg_mask: list[np.ndarray] = []
    raw_sessions: list[str] = []
    derived = bool(config["data"].get("emg_derived_channels", False))
    session_normalise = bool(
        config["data"].get("emg_session_normalise", False)
    )
    percentile = float(config["data"].get("emg_session_percentile", 95.0))
    log1p = bool(config["data"].get("emg_log1p", True))

    for row in tqdm(train.itertuples(index=False), total=len(train)):
        arrays = preprocess_grid_signals(row, config["data"], scaler=None)
        duration = float(arrays["duration_s"])
        if not np.isfinite(duration) or duration < minimum or duration > maximum:
            continue
        length = int(arrays["length"])
        count = min(args.samples_per_trial, length)
        indices = rng.choice(length, count, replace=False)
        imu = np.asarray(arrays["imu"])[indices].astype(np.float64)
        imu_mask = np.asarray(arrays["imu_mask"])[indices]
        imu[~imu_mask] = np.nan
        imu_values.append(imu)
        raw_emg.append(np.asarray(arrays["raw_emg"])[indices].astype(np.float64))
        raw_emg_mask.append(np.asarray(arrays["raw_emg_mask"])[indices])
        raw_sessions.append(str(row.participant_id))

    if not raw_emg or not imu_values:
        raise ValueError("No valid training samples were available for scaler fitting")

    # Pass 1: one gain reference per session, from that session's own training
    # amplitudes. A high percentile is the standard MVC-proxy reference.
    session_keys: list[str] = sorted(set(raw_sessions))
    session_reference = np.ones(
        (len(session_keys), raw_emg[0].shape[1]), dtype=np.float32
    )
    if session_normalise:
        for index, key in enumerate(session_keys):
            samples = [
                np.where(mask, values, np.nan)
                for values, mask, session in zip(raw_emg, raw_emg_mask, raw_sessions)
                if session == key
            ]
            stacked = np.concatenate(samples, axis=0)
            reference = np.nanpercentile(stacked, percentile, axis=0)
            session_reference[index] = np.where(
                np.isfinite(reference) & (reference > 1e-9), reference, 1.0
            )
    # Fallback for a session absent from training (an unseen participant at
    # test time): the median reference across the sessions we did observe.
    session_fallback = np.median(session_reference, axis=0).astype(np.float32)

    # Pass 2: fit the global robust statistics on exactly the representation
    # the fitted scaler will emit, so fitting and inference cannot diverge.
    reference_by_session = {key: session_reference[i] for i, key in enumerate(session_keys)}
    emg_values = []
    for values, mask, session in zip(raw_emg, raw_emg_mask, raw_sessions):
        features = raw_emg_features(
            values,
            mask,
            reference_by_session[session] if session_normalise else None,
            derived,
            log1p,
        ).astype(np.float64)
        full_mask = extend_emg_mask(mask) if derived else mask
        features[~full_mask] = np.nan
        emg_values.append(features)

    emg_center, emg_scale = robust_statistics(np.concatenate(emg_values, axis=0))
    imu_center, imu_scale = robust_statistics(np.concatenate(imu_values, axis=0))
    output = Path(args.output or config["paths"]["scaler"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        emg_center=emg_center,
        emg_scale=emg_scale,
        imu_center=imu_center,
        imu_scale=imu_scale,
        emg_log1p=np.asarray(log1p),
        emg_derived=np.asarray(derived),
        session_keys=np.asarray(session_keys if session_normalise else [], dtype="<U64"),
        session_reference=session_reference
        if session_normalise
        else np.zeros((0, raw_emg[0].shape[1]), dtype=np.float32),
        session_fallback=session_fallback,
    )
    print(
        f"Wrote grid scaler to {output.resolve()} "
        f"(EMG={len(emg_center)}, IMU={len(imu_center)}, "
        f"sessions={len(session_keys) if session_normalise else 0}, derived={derived})"
    )


if __name__ == "__main__":
    main()
