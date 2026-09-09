"""Load dense open/close labels onto the causal 100 Hz wearable grid."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_gripper_state(path, trial, max_gap_s=.02):
    frame = pd.read_csv(path, usecols=lambda name: name in {
        "time_perf_counter", "gripper_state"})
    if "gripper_state" not in frame or "time_perf_counter" not in frame:
        raise ValueError("missing per-timestep gripper_state or time_perf_counter column")
    time = pd.to_numeric(frame["time_perf_counter"], errors="coerce").to_numpy()
    state = frame["gripper_state"].astype("string").str.strip().str.lower()
    state = state.replace({"0": "open", "1": "close", "0.0": "open", "1.0": "close"})
    unknown = sorted(set(state.dropna()) - {"open", "close"})
    if unknown:
        raise ValueError("unknown gripper_state value(s): " + ", ".join(unknown[:5]))
    valid_time = np.isfinite(time)
    order = np.argsort(time[valid_time], kind="stable")
    time = time[valid_time][order]
    state = state[valid_time].iloc[order]
    keep = ~pd.Series(time).duplicated(keep="last").to_numpy()
    time, state = time[keep], state.iloc[keep]
    labelled = state.notna().to_numpy()
    if not labelled.any():
        raise ValueError("gripper_state has no open/close labels")
    label_time = time[labelled]
    label_value = state[labelled].map({"open": 0, "close": 1}).to_numpy(dtype=np.int64)
    grid = time[0] + np.asarray(trial["time"])
    index = np.searchsorted(label_time, grid, side="right") - 1
    found = index >= 0
    clipped = np.maximum(index, 0)
    valid = found & ((grid - label_time[clipped]) <= max_gap_s)
    labels = np.where(valid, label_value[clipped], 0).astype(np.int64)
    trial["gripper_state"] = labels
    trial["gripper_state_valid"] = valid
    trial["audit"]["gripper_state_valid_fraction"] = float(valid.mean())
    return trial
