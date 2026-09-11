#!/usr/bin/env python3
"""Same protocol as train_gripper_neuromuscular_future.py, for FFT-domain
trial CSVs whose timing columns are named freq_perf_counter/freq_s instead
of time_perf_counter/time_s(everything else about the schema is unchanged).

Reads the freq_ columns straight from disk (no renamed copies) by patching
pandas.read_csv inside reach_grasp/gripper_state for the duration of the run.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from emg_touch.data import gripper_state, reach_grasp  # noqa: F401 (share pandas module)
from emg_touch.data.click_target import CANVAS_COLUMNS, NORMALIZED_COLUMNS, PIXEL_COLUMNS
from scripts import train_gripper_neuromuscular_future as neuromuscular_future

assert reach_grasp.pd is gripper_state.pd is pd, "expected one shared pandas module"

RENAME = {"freq_perf_counter": "time_perf_counter", "freq_s": "time_s"}
# These are per-trial scalars, constant on every row, broadcast into the FFT
# the same as every other channel -- so they come back as FFT coefficients
# (~0 on most rows) instead of their real value. Restore them from the raw
# (non-fft) sibling CSV, which carries the same trial and still has them intact.
METADATA_COLUMNS = (*CANVAS_COLUMNS, *NORMALIZED_COLUMNS, *PIXEL_COLUMNS,
                     "sample_rate_hz_declared")


def _raw_sibling(path):
    path = Path(path)
    if path.parent.name != "fft":
        return None
    name = path.name
    name = name[:-len("_fft.csv")] + ".csv" if name.endswith("_fft.csv") else name
    raw = path.parent.parent / name
    return raw if raw.exists() else None


@functools.lru_cache(maxsize=None)
def _raw_metadata(raw_path):
    frame = pd.read_csv(raw_path, usecols=lambda n: n in METADATA_COLUMNS)
    return {col: frame[col].iloc[0] for col in METADATA_COLUMNS if col in frame and len(frame)}


def _patched_read_csv(original_read_csv):
    def read_csv(*args, **kwargs):
        usecols = kwargs.get("usecols")
        if callable(usecols):
            kwargs = {**kwargs, "usecols": lambda name: usecols(RENAME.get(name, name))}
        frame = original_read_csv(*args, **kwargs)
        frame = frame.rename(columns={k: v for k, v in RENAME.items() if k in frame.columns})
        path = args[0] if args else kwargs.get("filepath_or_buffer")
        raw = _raw_sibling(path) if path is not None else None
        if raw is not None:
            for column, value in _raw_metadata(raw).items():
                if column in frame.columns:
                    frame[column] = value
        return frame
    return read_csv


def _no_segmentation(t, mask, max_gap):
    # FFT windows are spaced far wider than the raw --gap-s threshold, so the
    # gap-based splitter in reach_grasp.segments() breaks every row into its
    # own segment and re-inits the filter state ~1300*28 times per trial.
    # Filtering is causal and stateless across calls only at segment
    # boundaries, so treating the whole trial as one segment is safe here.
    if len(t):
        yield 0, len(t)


def main():
    original_read_csv = pd.read_csv
    original_segments = reach_grasp.segments
    pd.read_csv = _patched_read_csv(original_read_csv)
    reach_grasp.segments = _no_segmentation
    try:
        neuromuscular_future.main()
    finally:
        pd.read_csv = original_read_csv
        reach_grasp.segments = original_segments


if __name__ == "__main__":
    main()
