#!/usr/bin/env python3
"""Summarize learned EMG-channel and IMU sensor/feature attention weights."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ATTENTION_PREFIXES = {
    "attention_imu_sensor_": "imu_sensor",
    "attention_imu_channel_": "imu_channel",
    "attention_emg_channel_": "emg_channel",
    # grid_crossvar: each is its own single-row softmax group, so the
    # per-prefix argmax below is valid for these three. The cv_<from>_to_<to>
    # columns are deliberately excluded: they are eight independent 8-way
    # softmaxes (one per "from" variate) flattened into 64 columns, and an
    # argmax across all of them would compare entries from different
    # distributions. Summarize those separately, e.g. by grouping columns on
    # their "from" prefix, or by reading cross_emg_to_imu/cross_imu_to_emg
    # directly.
    "variate_attention_": "variate",
    "scale_emg_": "patch_scale_emg",
    "scale_imu_": "patch_scale_imu",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--output", default="channel_attention_summary.csv")
    args = parser.parse_args()

    frame = pd.concat(
        [pd.read_csv(path) for path in args.predictions], ignore_index=True
    )
    if "requested_cutoff" not in frame:
        frame["requested_cutoff"] = "touch"
    required = {"model_kind", "configuration", "requested_cutoff", "trial_id"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Prediction files are missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    group_keys = ["model_kind", "configuration", "requested_cutoff"]
    for prefix, attention_type in ATTENTION_PREFIXES.items():
        columns = sorted(column for column in frame if column.startswith(prefix))
        if not columns:
            continue
        for keys, group in frame.groupby(group_keys, sort=True):
            available_columns = [column for column in columns if group[column].notna().any()]
            if not available_columns:
                continue
            values = group[available_columns].to_numpy(dtype=np.float64)
            # Rows from another modality can contain all NaNs after concatenation.
            valid_rows = np.isfinite(values).any(axis=1)
            values = values[valid_rows]
            if len(values) == 0:
                continue
            safe = np.where(np.isfinite(values), values, -np.inf)
            top_indices = np.argmax(safe, axis=1)
            for column_index, column in enumerate(available_columns):
                channel_values = values[:, column_index]
                channel_values = channel_values[np.isfinite(channel_values)]
                if len(channel_values) == 0:
                    continue
                rows.append(
                    {
                        "model_kind": keys[0],
                        "configuration": keys[1],
                        "requested_cutoff": keys[2],
                        "attention_type": attention_type,
                        "channel": column.removeprefix(prefix),
                        "trajectories": len(channel_values),
                        "mean_attention": float(np.mean(channel_values)),
                        "median_attention": float(np.median(channel_values)),
                        "std_attention": float(np.std(channel_values)),
                        "top_attention_fraction": float(
                            np.mean(top_indices == column_index)
                        ),
                    }
                )

    if not rows:
        raise ValueError("No attention columns were found in the prediction files")
    result = pd.DataFrame(rows).sort_values(
        [
            "model_kind",
            "configuration",
            "requested_cutoff",
            "attention_type",
            "mean_attention",
        ],
        ascending=[True, True, True, True, False],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
