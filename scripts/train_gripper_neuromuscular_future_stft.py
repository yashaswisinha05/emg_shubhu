#!/usr/bin/env python3
"""Same protocol as train_gripper_neuromuscular_future.py, but with EMG
features computed via a causal trailing-window STFT band-power
representation (emg_touch.data.reach_grasp.causal_stft_bands) instead of the
default trailing-RMS features.

Unlike train_gripper_neuromuscular_future_fft.py -- which feeds a *whole-trial*
FFT computed once per trial, so every timestep's "feature" already encodes
information from the entire future of the trial -- this variant computes a
strictly causal, trailing-only spectrogram: frame i uses only raw samples up
to and including i. It also keeps the EMG feature width identical to the RMS
baseline (one low-band + one high-band power per channel), so this is an
architecture-matched ablation: same model, same input shape, only the
per-channel feature computation changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import train_gripper_neuromuscular_future as neuromuscular_future


def main():
    if not any(a == "--emg-features" or a.startswith("--emg-features=")
               for a in sys.argv[1:]):
        sys.argv.extend(["--emg-features", "stft"])
    neuromuscular_future.main()


if __name__ == "__main__":
    main()
