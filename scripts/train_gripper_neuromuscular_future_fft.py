#!/usr/bin/env python3
"""Same protocol as train_gripper_neuromuscular_future.py, for FFT-domain
trial CSVs whose timing columns are named freq_perf_counter/freq_s instead
of time_perf_counter/time_s(everything else about the schema is unchanged).

Reads the freq_ columns straight from disk (no renamed copies) by patching
pandas.read_csv inside reach_grasp/gripper_state for the duration of the run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.data import gripper_state, reach_grasp  # noqa: F401 (share pandas module)
from scripts import train_gripper_neuromuscular_future as neuromuscular_future

assert reach_grasp.pd is gripper_state.pd is pd, "expected one shared pandas module"

RENAME = {"freq_perf_counter": "time_perf_counter", "freq_s": "time_s"}
# Not meaningful after an FFT transform (comes back as 0 or near-zero), and
# fails the raw-rate sanity check in reach_grasp.preprocess -- drop it so the
# check is skipped the same way it is for a file that never had this column.
DROP = ("sample_rate_hz_declared",)


def _patched_read_csv(original_read_csv):
    def read_csv(*args, **kwargs):
        usecols = kwargs.get("usecols")
        if callable(usecols):
            kwargs = {**kwargs, "usecols": lambda name: usecols(RENAME.get(name, name))}
        frame = original_read_csv(*args, **kwargs)
        frame = frame.rename(columns={k: v for k, v in RENAME.items() if k in frame.columns})
        drop = [c for c in DROP if c in frame.columns]
        return frame.drop(columns=drop) if drop else frame
    return read_csv


def main():
    original_read_csv = pd.read_csv
    pd.read_csv = _patched_read_csv(original_read_csv)
    try:
        neuromuscular_future.main()
    finally:
        pd.read_csv = original_read_csv


if __name__ == "__main__":
    main()
