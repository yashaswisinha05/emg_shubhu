"""Derive each trial's endpoint pose: where the reach actually finished.

The dense pose head answers "where is the hand right now". This is the target
for a different question -- "where is this reach going to end" -- which is one
SE(3) pose per trial, so it is predicted at every timestep and the useful
result is how early in the reach the prediction converges.

Taken as the componentwise median over a short trailing window of valid
samples rather than the single last sample, so one VIVE dropout or one noisy
frame at the end of a recording cannot define the target. Position and
orientation are resolved independently because their validity masks differ:
preprocess invalidates orientation on quaternion norm as well as on the
tracking-age and sync-error limits that gate position.
"""
from __future__ import annotations

import numpy as np


def _endpoint(values, valid, time, window_s):
    if not valid.any():
        return None
    cutoff = time[valid][-1] - window_s
    window = valid & (time >= cutoff)
    return np.median(values[window], axis=0).astype("float32")


def add_final_pose(trial, window_s=.1):
    time = np.asarray(trial["time"])
    position = _endpoint(trial["position"], np.asarray(trial["pose_valid"], dtype=bool),
                         time, window_s)
    orientation = _endpoint(trial["orientation"],
                            np.asarray(trial["orientation_valid"], dtype=bool), time, window_s)
    if position is None and orientation is None:
        raise ValueError("no valid VIVE samples, so the trial has no endpoint pose")
    if position is not None:
        trial["final_position"] = position
        trial["audit"]["final_position_m"] = position.tolist()
    if orientation is not None:
        trial["final_orientation"] = orientation
    trial["audit"]["final_pose_window_s"] = window_s
    return trial
